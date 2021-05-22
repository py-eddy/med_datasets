CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�E����     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N @�   max       P��n     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���m   max       ;�`B     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @Fg�z�H     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @vy�Q�     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q            �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �bN   max       %        ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��b   max       B1�     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0H�     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >f��   max       C��     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >B�   max       C���     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          X     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N @�   max       P��&     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?���e���     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       ;�`B     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @Fg�z�H     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vxQ��     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q            �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�!`         ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��O�M   max       ?��"��`B     p  a�               	               	      2   +                  =               /                                                      ,   8             
         (      !   	   W         W   +                  <       *            6               N[�OG�N�O�N��#N֘�O��N:+�O�ԋO��;N��N�VO�S�P>��N��UO5�N@u[OMqN|��P��nN:�^NP9�NW��Ow27O�Z�O�!�N�N$��N���O�U$O9��O FjO���Ou��O�OP�N.љN�QN��vN��oN�K�O�(�P
.XP&�NC�XN��O�oN�)N��N�[�O���N���O�pXN�@xP<�O 8�P��P,`�O�sO#��N~9NWueN�R�N&�UO�DlO�_9Oi��O_�NoZWO@<(O�Ol�N��N @�NP4�OD�o;�`B;ě�;�o%   ��o�o�D���D����`B�o�#�
�T���T���u��o��o��C����㼛�㼬1��1���ͼ��ͼ��ͼ��ͼ�/��/��`B��`B��h���������o�o�+�+�\)��w��w�#�
�#�
�#�
�0 Ž49X�49X�49X�8Q�8Q�8Q�@��@��H�9�H�9�L�ͽP�`�P�`�T���e`B�m�h�q���u��������7L��7L��C���C���O߽��P���-���-�������m������������������
#--#
�����HO[chpt{tnh[OMGEHHHH������������������������

����������������������������X[`gsgdghg[UTXXXXXXXcit��������������tuc�
/<BA><*#���������znca^abnz��������� ������������	�������������������������������FHKU`aefijaXUHBEFFFF��������������������#08<F<60%#������

������zz|��������������zzz���	#0<b{�����{T<������������������
 #%#
���������������������
/6<JAHH>0#
���')6BO\fnrth[OB6-'""'%*6CR\hp��uh\O6)BBOQSPOBA>BBBBBBBBBBNO[aa\[OIJNNNNNNNNNN�������������������)6BN[__[OF6)]amvz|���zmaa]ZYX[]!#/<HUW]agfaUH<</'#!'/5BN[gt����tg[B*#"'�������� ��������������������� �������FIOTamtvy|~~zmaTMHFF���������st|�����������tsssss��������������������X[grt{|tg[XUXXXXXXXX"",/8;<=>;;3/" """";Haz���}qjZTH?;81/3;�)/1BMMLB5�����)5B[jrwyvqf[N85.)&&)rt�����tmkrrrrrrrrrr���������������p��������������zuqp�����������������������������������������������������������)/5:COU\ans����zna/)������������������������
#/6:>A=<#
��������

����������kgaUN<5*&'.<Uaz����k26>BKOY[]hiztph[B722�����������������AAKJ@/#
������*0;A���������������������������������:<HHUadbaVUH<:::::::&)666/)|������������v||||||6)'")469666666666������%'$�����������������!)5BN[_c_RNB5)EIUbny{��������{naFE��������������������)/68;96) �)5=ISYZNG5)��������������������##(06;<:720.*&######/003<?B@<720.///////ntv�����|vtsrlnnnnnn������������������������غ޺�������������������y�t�l�g�b�`�g�j�t¡�e�]�^�e�j�r�|�~�������������~�r�e�e�e�e�H�G�<�8�/�+�/�6�<�H�U�a�e�d�a�`�W�U�H�H�׾ϾʾǾʾվ׾�����	��	� �����׾׾A�(���'�6�A�Z�s���������������s�Z�A�A�<�A�J�M�T�Z�d�f�g�p�f�Z�M�A�A�A�A�A�A�4�(�"���'�$�!�(�A�Z�l�w�v�e�X�W�M�A�4������.�4�B�[�t�w�t�n�r�h�O�B�6�)�����������������������������¿������������������'�)�5�A�B�B�G�H�B�=�5�,�)���-�%�$�/�?�G�T�`�y�����������y�m�`�T�;�-�y�`�V�U�`�����Ŀѿڿ�������ݿ����y�T�J�H�C�H�I�T�a�m�z�~�z�x�t�m�a�T�T�T�T���������������������������������������˻û����û˻лӻܻ�ܻۻлûûûûûûû����������������������������������������������������������������������������������������t�]�N�F�:�9�F�_�s���������������Ϲ͹ʹϹٹܹ�����ܹϹϹϹϹϹϹϹϼ����������������������������������������z�p�n�d�i�n�zÇÇÇÇ�}�z�z�z�z�z�z�z�zìàÙÓÒËÇÀÇÓßìù��������þùì�S�Q�J�H�E�F�S�l�x���������������x�l�_�S��	�����������	���"�/�;�>�?�=�5�.�"��`�^�Z�`�m�y���y�m�`�`�`�`�`�`�`�`�`�`�.�-�%�.�;�G�I�P�G�;�.�.�.�.�.�.�.�.�.�.�����������|�~���������������������������������{�}������������������������������ĳĨĭĿ�������������
�������������Ŀĳ����������������������
������������׾оɾɾѾܾ����	������	��������s�g�`�a�c�g�n�s�����������������������f�^�Y�P�M�I�D�M�O�Y�]�f�r�s�w���r�o�f����ƳƬƦƪƳ�����������������������������������
�����������������������������¿¼¿�����������������������������˺Y�X�L�E�E�L�Y�e�r�~���~�s�r�e�Y�Y�Y�Y�Y�z�r�s�z�|���������������z�z�z�z�z�z�z�z����������������������������������������������������(�N�c�z�����g�N�5��������������� �;�T�a�i�j�^�X�M�;�/�"�	���s�`�V�e�s�����������������������������s�������������������������������������������	���������	���"�-�.�6�.�"������������3�<�@�L�Y�\�[�Z�U�J�@�3����'��'�*�4�:�@�D�M�Y�f�l�k�f�d�Y�M�@�4�'�x�w�l�i�d�h�l�n�x���������������x�x�x�x����������������������������������������FFE�E�E�E�E�E�E�E�E�E�E�E�FFF#F%FF���������������ʼּܼݼ׼ּʼ������������ݿѿĿ������ÿѿݿ������������ݺɺú��������ɺֺߺ����ֺɺɺɺɺɺɿ.�G�O�E�;�&���׾ʾ����������������.�ܹ۹Ϲù��������������ùƹϹٹ������{�n�eŊŔŠũŹ�������������ŭŠŇ�{D�D�D�EEEE+EBEJECE+ED�D�D�D�D�D�D{D����������Ƽ�������������ּ�����ŭŬŠŔŊőŔŠŭŹ����������������Źŭ�g�d�Z�Y�T�Y�Z�g�g�s�t�z�x�s�g�g�g�g�g�gùðìëìîøùú��������ýùùùùùù�I�H�?�B�I�S�V�X�b�k�k�b�]�V�I�I�I�I�I�I���������������������������������������������ùàØÔØàì��������������!����)�)�9�G�S�l���������y�k�g�P�(�!�{�o�b�V�O�I�C�>�<�=�F�I�V�b�o�q�w�{�~�{�-�)�%�%�'�-�1�F�S�_�l�x�������{�l�_�:�-�����������������������������������������������ĽȽҽڽݽ��ݽнĽ����h�[�S�O�O�V�[�h�tāčĔĚĠĢĞĕā�t�h��������������ùìëääçìù���������߽��������������Ľнݽ�ݽнĽ���������������������(�-�(�����������������N�B�B�@�B�N�[�e�g�t�}�t�g�[�N�N�N�N�N�N����������������������� �������������� ` C 0 3 ) G i c d F T > A 7  ? ; T b 0 : , G " _ F Q X ) c E F 5 J . ? < ` ) m | @ 3 B c R d ` g B ( : 5 6 \ � m W X T j 0 3 F X A n , 4 / 1 l � n <    s  �     �  �  �  p  R  �    +  @  �  �    k  ,  �  F  Q  n  n    >  l    /  �  �  �  |  �    x  �  ]  �  �  �  �  @  �  x  F  �  d  =  �  �  �  �  �  �  ^  �  &  �  3  y  �  �  �  F  E  :  �  v  |  �  ]  1  �  v  �  �%   ��9X�u�u�t���9X���
��9X��/��C���1�}�aG���9X�ě����
�o���ͽ��
��`B�ě����]/��t��,1��h���+�ixսD���8Q�u�aG��,1�T���t��t����''y�#��1�ě��<j�]/����Y��T���D����{�y�#�����e`B�C����㽝�-�I���vɽq���y�#��\)��t�����   �Ƨ��;d�� Ž��P�������m�\��Q콧���bNB��B��B6zB! �B#��B!�B�VBr�B��B<B�B�%B+�B�bB3B%��B}�B	B&y�B)1B$�MB �PB��BuB1�B�tB	nB�@B�LA�khB0B��B�(B"�KA��fB ��B�B"��B	AA��bA�0?BnB�B��B��B�B��B l�B�2BS�B a�B]�B#��B�]B3�B�zB��B-T�B|"By�B�#B
ݍBI�BYB�Br�B'��B��B�KB�BB(�B%ǀB&lB	�bB�)B�'B��B:�B �<B#�*B!��B�BN}B��B�<B�jB��B+9�B��B0�B%�dBH�B��B'�B>jB$�B ��B��BErB0H�BjBtBX,B�uA�hB�XB��BC�B"�yA�eB �B��B"@�B	>�A���A�q�B<ZB?�B��B��B@ B�wB RsB��B�mB B�B�B$(�B�)B@�B�B2cB-A�B@�BZBίB
�aB?HB@�B�B=�B'�LBôB.tB�FB��B%��B&DB	�wB!@NF�A�n�?�l'A��DAU��A@_\A>-�A;� A� kA�,A��Ag�rAvYrA��@A���@��MA�xLA�Q9A�-�>�3�@��A�=�A̩8@�"�A^)$Ak!�Ac#vA��SA���A�A�LPAX��A�s�@�>BL�B��A�n�?�yeA��9A���A���A�A���A"�<A]@1?��@�Hd@��vA�J�C��@�GA�;@8�AU��>f��A�4�C�IlA-mA��nA�~�Aͩ�B��A�~\AώYA*�B�@���A/k3A%��A�ϝA�^A&
�A2��A��#A�w�@K:�A�y�?��oA��AU1mAAv5A?��A;��A؇�A��A�K�Ah��At�:A���A��@��A�2;A��A���>�b�@�[�A�}�A�{r@��A\;Ak�Ab�A�̚A�y�A��A��$AX�xA�m�@ݺ�B>�B��A���?Ӧ�A���A�VhA��FA�~�A�x�A"m�A\�F?��@�X>@�F�A��{C���@��pA~,q@4 �AVԹ>B�A���C�XDA��A�NA��À+B�A�{^Aπ�AB>:@{֞A0�A$�HAܒ�ÀbA$*�A5�A��A�}V               	               
      2   +      	            >               0                                                      -   9         !   
   	      )      !   
   X         W   +                  =   !   +            7                                 #      %   !         '   3                  A                  !                                                +   #   '                     #            ,      1   6   '                  !   %                                                         !            /                  ;                  !                                                +   #   !                                       +   +   %                     #                              N[�O�_N�P�N���N�$O
?N:+�OO��O��;N��NąO��AP3$N��UO5�N@u[N떞N*��P��&N:�^NP9�NW��O�O�TO�!�N�N$��N���O�0fO9��O FjO���Od[kO�O.�(N.љN�QN��vN��oN�K�O�(�P3O���NC�XN��O�mN�)N��N�[�O�+�N���O�pXN�@xO��N(#O�)'O���O��MO#��N~9N�N�R�N&�UOò�O���Oi��O_�NoZWO@<(O��Ol�N��N @�NP4�O7��  !  �  �  �    �  B    �  �  b  |  �  �  .  6  �  �    �  =  �  �  �  C  �    P  E  �  �  G  �  T  w  �    S  7  #  ;  	  �  �  3  �  8  ~  �  h  �  r  �  	8  �  �  ]  �  �  �  �  �  �  
�  �  	)    <    
I  	  0  �  �  f;�`B;o;o�o�D���49X�D���#�
��`B�o�49X��`B�u�u��o��o��t����
��1��1��1���ͼ��,1���ͼ�/��/��`B����h���C��o���C��o�+�+�\)��w��w�,1�@��#�
�0 Ž<j�49X�49X�8Q�]/�<j�@��@�����}�]/��o�Y��T���e`B�u�q���u��O߽�7L��7L��7L��C���C����P���P���-���-�����������������������
#**#
������JO[bhnskh[ONHFJJJJJJ�����������������������


�������������������������������X[`gsgdghg[UTXXXXXXX���������������������
/<BA><*#���������znca^abnz��������������������������������������������������������FHKU`aefijaXUHBEFFFF��������������������#08<F<60%#����

�����������������|�	#<b{�����{U<#
�������������������
 #%#
�������������������� 

#-/0<>B<8/+#
 -69BOU[^ab^[OBB65/--%*6CR\hp��uh\O6)BBOQSPOBA>BBBBBBBBBBNO[aa\[OIJNNNNNNNNNN�������������������)6BLZ^][OB6)	]amvz|���zmaa]ZYX[]!#/<HUW]agfaUH<</'#!&)5BN[]gt}���xtg[B/&������������������������������� �������GHJPTamrtwyxmaTOHGGG���������st|�����������tsssss��������������������X[grt{|tg[XUXXXXXXXX"",/8;<=>;;3/" """";Haz���}qjZTH?;81/3;�).BLKIB5������.5BN[gmrtrm[NB50+)*.rt�����tmkrrrrrrrrrr���������������u{���������������{yu�����������������������������������������������������������;?GHUanz����zvnaUL<;������������������������
#/6:>A=<#
��������

����������5<HUamuxxqlaUH<73215KO[[\hmhf[OLKKKKKKKK����������������������#+/:AA>#
��������������������������������������:<HHUadbaVUH<:::::::!)4,)|������������v||||||6)'")469666666666���"%%#��������������������!)5BN[_c_RNB5)EIUbny{��������{naFE��������������������)/68;96) �
)5:FORO@5)
��������������������##(06;<:720.*&######/003<?B@<720.///////ntv�����|vtsrlnnnnnn������������������������غ޺�������������������t�o�i�g�e�d�g�m�t�t�e�_�`�e�l�r�~�����������~�r�e�e�e�e�e�e�U�T�H�<�;�/�-�/�;�<�H�U�a�c�a�a�^�U�U�U�׾Ӿ̾׾ݾ�������������׾׾׾׾׾׾M�K�A�<�A�E�M�Z�f�s�~�������t�s�f�Z�M�A�<�A�J�M�T�Z�d�f�g�p�f�Z�M�A�A�A�A�A�A�4�.�1�+�*�-�4�8�A�M�Z�^�h�f�b�Z�Q�M�A�4������.�4�B�[�t�w�t�n�r�h�O�B�6�)�����������������������������¿������������������)�)�5�B�E�G�B�<�5�+�)�����T�G�;�7�4�9�G�K�T�`�p�y�������y�s�m�`�T�y�a�X�W�`�������Ŀѿؿ޿���	��ݿ����y�T�J�H�C�H�I�T�a�m�z�~�z�x�t�m�a�T�T�T�T���������������������������������������˻û����û˻лӻܻ�ܻۻлûûûûûûû������������������������������������������������������������������������������������v�_�H�<�>�K�d���������������������׹Ϲ͹ʹϹٹܹ�����ܹϹϹϹϹϹϹϹϼ����������������������������������������z�p�n�d�i�n�zÇÇÇÇ�}�z�z�z�z�z�z�z�zìàà×ÖÓÓÓÜàìóù����������ùì�_�T�S�P�P�S�Y�_�l�x���������������x�l�_��	�����������	���"�/�;�>�?�=�5�.�"��`�^�Z�`�m�y���y�m�`�`�`�`�`�`�`�`�`�`�.�-�%�.�;�G�I�P�G�;�.�.�.�.�.�.�.�.�.�.�����������|�~���������������������������������}�~������������������������������ĳĨĭĿ�������������
�������������Ŀĳ����������������������
�������������־;̾Ծ׾߾����	�������	���s�h�a�`�a�d�g�r�s���������������������s�f�^�Y�P�M�I�D�M�O�Y�]�f�r�s�w���r�o�f������ƳƮƨƬƳ���������������������������������
�����������������������������¿¼¿�����������������������������˺Y�X�L�E�E�L�Y�e�r�~���~�s�r�e�Y�Y�Y�Y�Y�z�r�s�z�|���������������z�z�z�z�z�z�z�z����������������������������������������������������(�N�c�z�����g�N�5�������������"�;�T�a�g�h�]�V�L�;�/�"��	���s�i�^�c�l�x���������������������������s�������������������������������������������	���������	���"�-�.�6�.�"��������� ��� �3�@�L�Y�Z�X�X�S�H�@�3�'��'��'�*�4�:�@�D�M�Y�f�l�k�f�d�Y�M�@�4�'�x�w�l�i�d�h�l�n�x���������������x�x�x�x����������������������������������������E�E�E�E�E�E�E�E�E�E�FFFFFF FFFE����������������ʼּۼּܼռʼ������������ݿѿĿ������ÿѿݿ������������ݺɺú��������ɺֺߺ����ֺɺɺɺɺɺɾʾ����������ʾ׾���	�� �������׾ʹù����������ù͹ϹֹҹϹùùùùùùù��n�i�uŇŔŠŬŷ������������ŭŠŇ�{�nD�D�D�D�D�D�D�D�D�D�D�EEE:EEE?E&EED󼘼������ȼ��������������ּ���ŭŬŠŔŊőŔŠŭŹ����������������Źŭ�g�d�Z�Y�T�Y�Z�g�g�s�t�z�x�s�g�g�g�g�g�gùöìëìðù��������úùùùùùùùù�I�H�?�B�I�S�V�X�b�k�k�b�]�V�I�I�I�I�I�I����������������������������������������çàÛÖÚàì����������
�	�
�����ùç�����"�+�,�=�G�S�`�y�����y�i�f�O�%��{�o�b�V�O�I�C�>�<�=�F�I�V�b�o�q�w�{�~�{�-�)�%�%�'�-�1�F�S�_�l�x�������{�l�_�:�-�����������������������������������������������ĽȽҽڽݽ��ݽнĽ����[�U�R�R�X�[�h�tāčĚĞğěđčā�t�h�[��������������ùìëääçìù���������߽��������������Ľнݽ�ݽнĽ���������������������(�-�(�����������������N�B�B�@�B�N�[�e�g�t�}�t�g�[�N�N�N�N�N�N�������������������������������������� ` @ , 8 1 - i G d F Q 8 : 7  ? + O U 0 : , ;  _ F Q X & c E K 0 J & ? < ` ) m | @ ' B c Q d ` g ' # : 5 # = z V M X T ] 0 3 : Z A n , 4 ( 1 l � n 4�  s  a  �  �  �  3  p  �  �    �  ;  G  �    k  �  ]  �  Q  n  n  Y  H  l    /  �  \  �  |  e  �  x  k  ]  �  �  �  �  @  m  �  F  �  #  =  �  �    �  �  �  h  L  m  �  �  y  �  Y  �  F  �  �  �  v  |  �    1  �  v  �  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  !        
    �  �  �  �  �  �  h  M  0    �  �  �  �  �  �  �  �  �  �  �  r  A    �  �  e  *  �  �  H  �  �  -  �  �  �  �  �  �  �  z  a  @    �  �  �  p  F    �  0  �  �  �  �  �  �  w  `  G  ,    �  �  �    ^  8    �  �  �  �  �  �  �       �  �  �  �  �  �  �  �  �  �  m  D     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ]  '  �  �  6  B  =  7  1  ,  &                 �   �   �   �   �   �   �   �  �  �  �        �        �  �  �  �  �  Q  )  �  �    �  �  �  �  �  r  L  %    �  �  �  �  o  3  �  �  {  =    �  v  c  T  F  6  &      �  �  �  �  �  �  u  W  2  �  r  L  Y  `  W  L  @  4  &    �  �  �  �  o  W  s  r  N    �  �  �  "  H  h  x  |  t  _  @    �  �  T  �  �    �  /  x  �  �  �  �  �  �  �  �  n  E    �  �  n  "  �  }    �    �  �  �  �  �  �  �  v  \  ?       �  �  �  �  s  H     �  .  )  %        �  �  �  �  �  �  �  �  �  �  �  ~  q  e  6  8  9  ;  <  ;  3  ,  $        �  �  �  �  �  �  �    �  �  �  �  �  �  �  j  P  4    �  �  �    \  8  A  �    r  x    �  �  ~  y  s  l  d  ^  W  Q  N  P  S  U  W  Y  [  �     �  �  �  �  i  =    �  �  ~  A    �  {    �     �  �  �  �  �  �  �  �  {  m  W  ?  &    �  �  �  u  L  #   �  =  8  3  /  *  %  !            	    �  �  �  �  �  �  �  �  �  �  l  F    �  �  �  r  C    �  �  �  X  )  �  �  !  q  �  �  �  �  �  �  ~  _  5    �  �  p      �  7  �  P  }  �  �  �  �  �  �  �  �  �  �  q  6  �  k  �  \  �  �  C  7  +      �  �  �  �  �  �  �  {  _  <    �  �  v  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  P  J  E  ?  9  0  '      
  �  �  �  �  �  �  �  �  s  X  :  C  D  ;  .        �  �  �  �  �  f  D    �  ~  �  A  �  �  �  �  �  �  �  �  |  c  E  "  �  �  \    �  c  �  a  �  �  �  }  n  Z  A  '    �  �  �  �  R    �  �  F  �  �  *  B  G  <  $  �  �  �  c  9       �  �  �  �  U  �  �   �  z    v  i  _  R  >  )    �  �  �  �  _  2    �  �  �  �  T  G  :  -      �  �  �  �  �  �  l  L  *  �  �  �  �  q  P  s  v  l  _  L  5      �  �  �  e  2  �  �  �  L  �  �  �  �  �  |  t  l  c  Y  O  E  9  *      �  �  �  �  �  �                      
       �   �   �   �   �   ~   e  S  G  :  .       �  �  �  �  �  �  �  �  y  l  o  u  |  �  7  /  &        �  �  �  �  �  �  �  �  d  D     �  �  {  #        	    �  �  �  �  �  �  �  �  �  �  �  �  �  v  ;  3      �  �  �  �  �  �  �  ~  N    �  �  _    �  ^    	    �  �  �  �  �  t  C    �  d    �    �  ;  �  �  �  �  �  �  �  �  �  �  �  ~  [  .  �  �  Z    �  R  �  O  �  �  �  �  �  �  �  o  ]  J  6      �  �  �  �  �  b  B  3  %      �  �  �  �  a  ?  !  �  �  �  N    �  �  f  )  �  �  �  �  �  �  �  �  �  �  �  �  g  -  �  %  �  )  �   �  8  #      �  �  �  �  �  �  �  x  X  5    �  �  �  �  i  ~  s  i  b  ^  Z  V  S  N  H  A  9  /      �  �  �  B  �  �  �  y  p  f  ]  S  G  9  ,        �  �  �  �  �  �  �  �    $  G  b  d  T  8    �  �  _    �  '  �  �       �  �  �  �  �  �  �  �  �  w  ]  @    �  �  �  O  �  �  :  �  r  m  k  o  _  E  )      �  �  �  x  4  �  �  K  �  �  �  �  �  �  �  �  w  h  X  G  4      �  �  �  �  `  9  �  y  �  �  �  �  	   	  	/  	8  	6  	-  	  �  �  f  �  B  d  S  �  &  �  /  |  �    >  d  �  �  �  �  �  m  E  	  �  I  �  0  �  �  �  �  �  �  �  �  �  �  �  �  �  a    �  �  ;  �  �  �  �  w  '  W    �  L  �  r  
�  �  K  
�  
N  	�  �  �  6    �  e  �  �  n  P  ,    �  �  �  G    �  |  +  �  V  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  o  _  N  =  �  �  �  �  �  �  �  �  �  �    l  Y  F  3          	  �  �  �  �  �  �  �  �  I    �  �  O  	  �  j    �  j    �  �  �  �  �  �  �  e  E  $    �  �  �  n  ?    �  �  x  �  �  z  Z  C    �  �  �  �  `  ,  �  �  O    �  l  "  �  
�  
�  
�  
�  
�  
�  
T  
  	�  	o  		  �  �  >  �  �  �  �  �  I  �  �  �  �  �  �  �  Y  &  �  �  b  	  �  *  �  3  �  �  L  	)  	  	  �  �  �  �  Z  *  �  �  Y  �  �    �  �     �  �      �  �  �  �  x  T  ;    �  �  u  3  �  �  z  �  c    <  9  6  3  $      �  �  �  �  �  �  z  p  e  [  P  D  9          �  �  �  �  �  �  �  j  M  0    �  �  �  _    
  
=  
I  
2  
  	�  	�  	�  	b  	!  �  m    �  �  ^  �  �  �  �  	  �  �  �  �  h  D  &    �  �  �  h  0  �  �  -  �  +  �  0  #    �  �  �  q  ?    �  �  b  (  �  �  t  6  �  �  �  �  �  �        �  �  �  �  �  �  ]  1    �  �  r  >  	  �  �  �  �  �  �  �  �  �  �  �  �  �  v  i  ]  P  D  7  +  a  f  a  V  E  ,    �  �  �  X  $  �  �  y  8  �  �  �  
