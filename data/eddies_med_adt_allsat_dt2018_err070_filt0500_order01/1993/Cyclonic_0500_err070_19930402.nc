CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�n��O�<     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�A�   max       P�#�     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��Q�   max       =o     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F7
=p��     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @v\�����     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��S�   max       <���     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��O   max       B/�V     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�5   max       B/�:     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��   max       C���     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >,,�   max       C��     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          H     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          A     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          9     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�A�   max       PmC�     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����l�D   max       ?�oiDg8     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��^5   max       =o     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F&ffffg     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �θQ�     max       @v\�����     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q�           �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�߀         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?nOv_ح�   max       ?И_��G     �  b�      &               !      #                           #   H      !            C                           $      !                  %      	         
   "   0                  $            8                                                   N��LOO��N+<�O��N�N���O7��N��.P�N���N\y�N�<+O*�N&�N���O�\fN��OڕP<�\NB�"P1x�O�,�O���O+%#PP��O[��NriN��5OI�O|ZNi��O܎O�!�O�OP�P
2PO-k'M�A�N N��!N)/�P7�#O�kN�RO�S�O9�^O .TO�/%P�#�N��OЏ4NϿ�N��oOn %OrՎOU=_N�C�N=�oO��
N���O�ȧNٌ;OtnNѰN��\N�"8N�;�N��O�2�OS��O
n[NV��N���OM�N�G�N�#�=o<T��<49X<t�<t�;ě�;��
;D����o�o��o��o�ě��o�#�
�49X�e`B�e`B�e`B�u�u��o��o��o��C����㼣�
���
���
���
��j��j�ě��ě��ě��ě����ͼ�����/��/��`B��`B�����o�o�o�o�C��C��\)�t��t��t���P�#�
�#�
�',1�,1�<j�<j�D���D���H�9�H�9�P�`�P�`�T���aG��aG��aG��q�������Q콸Q����������������������)*-+)'�����7BEOQVOB=67777777777��������"%
������)6BEHHCB6,)#!�������������������
�����057BNRXYRNB531000000���������
���������
"#%#
���������
����������#&/<EHMLHC=<;//,+#^hsty���������sih^\^���������|��������������������������������������������������TTT[^aefgiijaZTTTTTT��������������������~����������������}y~�����������������������#0IUnyyqb<0#�������������������������������}�%)6BHLWZ[_[OB<6)z���������������uqsz#0<EIJUY[XUQI<0$��������������������#/<DG<5/#��������������������q{������������{yrrqqaafnz������znaaaaaaa����������������xwBN[gty������tg[NB69B��������������{uzz���
##
������������<FG#�������HRUanz�������zna`VHH��������������������lmz|~znmmlllllllllll������������������������������������������������# �������))+*)'!  #),46886)$rw�������������}ylpr�������

��������]aenz~������zna`_^]]*6CDOW[XPC6*����6\g[6��������otw����������tloooo���������������������������������������� "" 	  �������������������������'/0-)���������������������������������������������55;BN[WNHB=555555555)6BO^iqtr[OB:)v{�����������{xuvvvv[`gu������������tcZ[����

 �������������������������������������������������������������������������������������������������������������������������������4<IXnsy{zrnbUID<9824���%)+*'!�����().5BENQSQNJB?5--,)(yz{��������zyyyyyyyy#09<70.#;<Uanyz��}zunaUHGC@;��

	���������������
 !#$#
� ����/�$�#����#�*�/�<�D�H�R�H�>�<�/�/�/�/�ֺ˺ɺƺȺɺպֺ����������������U�Q�U�^�a�n�x�r�n�a�U�U�U�U�U�U�U�U�U�U�4�(��
���� �4�E�M�Z�f�j�y�r�s�p�M�4������#�/�<�=�D�<�9�/�#�����������������������������������������������z�t�n�t�¦²·¹·²¦�������y�~���������������������������������������������������$�0�7�=�?�<�0�����������������������������������������������������������������������������������B�<�6�3�4�6�8�B�O�X�h�t�u�t�i�h�[�O�F�B�(�"�����(�+�A�M�f�k�g�f�Z�M�L�A�4�(�������������������������������������������"�%�&�(�������������v�|���������ʾ׾������׾ʾ�������������Ƶ����������������������������	���������5�A�Z�g�m�s�o�g�Z�A�(��	�M�4����,�D�M�f�r��������������f�Y�M����������������������������������������~�n�h�d�f�w�������������������׿	���׾ξξ׾���	���"�1�:�=�7�.��	��óñëÜ×Øàì���������������������ſ`�T�P�J�I�T�`�m�y���������������}�y�m�`�����U�G�>�@�L�e�~�������� �
��ɺ������������������(�4�;�A�H�M�=�(����6�,�*������*�6�C�O�Q�O�C�C�6�6�6�6�H�B�@�E�H�U�[�a�e�h�a�U�H�H�H�H�H�H�H�H�	��������������"�/�7�;�F�K�H�;�/�"��	�������������Ľнݽ�������ݽнĽ����U�U�H�F�C�A�G�H�P�U�Z�[�Z�U�U�U�U�U�U�U�������~���|�������Ŀݿ����� ����ѿĿ����ھؾݾ����	���������	����ŠŔņŁŁřŪŭŹ������������������ŹŠŭūũţŧŭŶŻ��������������������Źŭ�Ľ��������������������Ľ��������ݽ������������������������������������������a�n�x�t�n�d�a�`�\�a�a�a�a�a�a�a�a�a�a�a�A�:�5�A�N�W�Z�[�Z�N�A�A�A�A�A�A�A�A�A�A���������������ĿͿϿĿ������������������l�i�l�o�x�������������x�l�l�l�l�l�l�l�l�����r�r�|����������.�4�2����޼ʼ����Z�T�G�H�M�M�Z�f�k�s�������������s�f�Z���������������þʾ׾ھھ׾Ҿʾ���������ĦĚă�|�|čĦĿ������������������ĿĳĦ����������������������� �
����
�������T�Q�H�>�B�H�T�a�b�m�t�z�����|�z�m�a�T�T��	����������	��"�.�5�8�:�8�.�"��:�!� �0�G�l���������ͽ�����ݽĽ��y�S�:������������������������������������������̾žþǾʾ�����"�.�;�F�C�;�"�	���ā�}�u�wāčĚěĦİİĦĚčāāāāāā�5�(�(���������������(�(�1�5�?�5�5�L�I�E�>�=�@�J�Y�e�r�~���������r�e�]�Y�L�Y�O�Y�b�f�z�����������ļƼż������r�f�Y�b�U�C�<�9�C�I�U�b�n�{ŁņŊŇņń�{�n�bŠŝŘŔŇŁŇňŔŠŭŰŵŴŭŢŠŠŠŠ�Ľ½��������ĽʽнҽнŽĽĽĽĽĽĽĽĻ|�q�e�`�a�l�x�������������������������|�������'�2�4�:�@�L�@�9�4�'����¿¦¨ª²¿������������������¿�ɺź����������ɺֺ׺��������ֺɺɻ����ܻ׻ܻ������'�2�/�'� ������û������������������ûȻûûûûûûû��6�,�)� ��)�+�6�?�B�F�O�O�O�C�B�6�6�6�6������������������������������������������������������������������������������������������� �����#����������������!���!�-�:�S�_�l�������������q�_�F�-�!��������������������������!�������ÓÒÇÀÄÇÑÓàìñù��������ùìàÓ���������������ĿǿǿĿ����������������������������ûлٻܻ��ܻлû����������������������������ùϹعܹ޹޹عǹù�����D�D�D�D�D�EEEEEED�D�D�D�D�D�D�D�D�E*EE*E-E7E>ECEPE\EbEiEtEqEmEiE\EPECE;E* D ( Y 8 3 + 8  - + 9 D < 1 3 / �  3 W ; G w K k % x D R ) y ' F 4 0 f J W d b U _ 1 a ( @ 0 < \ 8 9 B l 3 o . 5 J 5 J > D D ` # 3 $ G ; @ 8 * ( E N i  �  �  \  �  �  �  �  �  E  �  f    �  ,    G  �  �  ;  c    ]  g  �  {  �  �  �  �  %  �  �     6  N    �  ,      d  �  0     �  �    Q  �  �  �  �    �  �  �  �  j  �  �  �  �  j  L  �  �  �  �  �  �  5  g  �  �  �  <<�������:�o�T��;�o;o��/�o�t��t���`B��o�ě��D����j��㼓t��D����{��/�@��<j�\)��/����0 ż�j�+�'����Y��49X�u�C��ixս��+��h�C��\)����49X�#�
�aG��49X�,1������T�<j��+�H�9�<j�]/����q���P�`�0 ŽƧ�D������ixս�+�P�`�y�#�y�#�ixսe`B���w��hs��O߽����hs�Ƨ��S���G�B�B@�BO}B��B��B.a�B��B�GB�B$rPB=�B�$B�BߘB��B!\A��OBCB �VB=yB&B�$Bk�B�pB�gB&N8Bc/B�B!~5B)f�B<!B*dB	�B�B�wB"��B�eB"MPA�"�B��BqEB,��Bu�B#BN�B�.B��B/�VB}�B�kB��B��B��B!}BB)�B6B��B��B_�B)'�B
[ZB#��B��B�"By�B�B�1B��B'1ZB�!B}�B�wB%T:B4�Bk-B�lB��BD7BBHBJlB��B.F,B�~B��B@�B$��B8�B�B��BMB�B!?fA�5B.�B �lB@�B&|1B��Bt�B=BCwB&��B>�B�B!ANB)��BYB*D�B	�B�]B��B">�B�
B"?�A�QGB��BC�B.>�BBaB��B6�B=B�{B/�:B�2B��B�yB��B�0B!��B=`B@B��B�XB��B)?�B
+B$:SB8�B��BB�B?
B��B�8B'?�B�mB�B̺B%<�B)�B;BB��A·N@H�AƣzA:�AA��2A!vA��JAq��B��@�;AI�BA�]�A:��@�A�Aԁ�ANu|BTA��x@ۈ�A�A��TA[(A�FAke�@>��A5��B �A�I4A�>*A(�*A�>�Ax>4AYc�A���A��[A(R3A�s�A�.�A�?.Av��@��dA&1AA}�AN2CA�}�A��A���A\b�A��A ��AX�YA��A���?��@�4XA��5A�J]A&a3@��'@���A��@7ʛ@�!�@��(A׋�A�K�A�4�B��@���A��A��Av��@��=��C�I�C���A+@E:�AƂ�A:��A�k�A�A���ArB	?�@���AH� A�@�A:�@또A��AP��B�yA��/@ېA�PA��A\��A��Ak;(@<��A6ƴB �5A�n�A�|�A*�1Aą�Az�AY�dA���A��VA*�A�wuA��=A�r�Aw$@�A�AB��AO�A��bA暔A��9A[��A }bA! �AV��A�e(A�m�?�uy@�3A�}:A��A&��@��&@��xA�t�@4�@���@�JGAפ>A�T�A���Ba)@~[�AўCA���Av� @�w�>,,�C�@pC��      '               !      $                           #   H      !            C                           %      "                  &      
            "   0                   %            8                                                               %               %                     %         1      -            7                     %      #      )      
            7         #            A      !                              %                        !                                                %                     !         +      '                                 %                  
            1         !            9                                    %                                             N��LO�N+<�N��BN�N���N�lN��.O�m3N���N\y�N�6�N�fN&�N�;�O��N��O;�xPz`NM�O�p`OK��O���O+%#O8>?O(�lNriN=w�O��N��Ni��O܎Oq�OX��OP�O��BO-k'M�A�N N��!N)/�P �O�kN�RO���O9�^O .TO2�PmC�N��O���NϿ�N�1�On %O[< OU=_N�C�N=�oO��N���O�ȧN���O9�NѰN��\N�"8N�;�N��O��N��(O
n[NV��N���O@�]Nt�]N�#�  �  z  �  F  �  �  �  7  E  �  K  �  �  �  &    �  d  l  �  �    f  �  �  Y    �    �  c  >  F  �  v  x  �  '  :  s  d    D  �  �  �      b  �  |  �  �    �  �  �  Y  �  Z  �  �  �    �  :  �  �  �  �    �  �  v  
  �=o;�`B<49X��o<t�;ě��o;D���o�o��o�ě��#�
�o�D����o�e`B��h��/��o��9X��9X��o��o�aG���j���
��j�ě��ě���j��j�����ě���P���ͼ�����/��/��`B�������\)�o�o�,1���C��,1�t���P�t���w�#�
�#�
�'@��,1�<j�@��H�9�D���H�9�H�9�P�`�P�`�]/�}�aG��aG��q�����P��^5��Q����������������������')+)# ����7BEOQVOB=67777777777�����������������)6BEHHCB6,)#!���������������������������057BNRXYRNB531000000��������� ����������
"#%#
���������
����������+/<?HKJHE@<3/..#++++ght���������}tphgcgg���������|��������������������������������������������������TTT[^aefgiijaZTTTTTT����������������������������������������������������������� 
#0IUbnrqhUI<0�������
������������������������}�%)6BHLWZ[_[OB<6)��������������������##'0<IOUWXVUMI<60'##��������������������#/9<><0/#!��������������������w{{����������~{xwwwwaafnz������znaaaaaaa����������������xwIN[gty����ytg\[NDCII�����������������������
##
�������������
		�������HRUanz�������zna`VHH��������������������lmz|~znmmlllllllllll���������������������������������������������!���������))+*)'!  #),46886)$tz��������������tpt�������

��������]aenz~������zna`_^]]*6CGMNJC96*���6HPOA���������otw����������tloooo����������������������������������������!! 	�����������������������%./.,)���������������������������������������������55;BN[WNHB=555555555%)06BO]goqo[OA6v{�����������{xuvvvv[`gu������������tcZ[����

���������������������������������������������������������������������������������������������������������������������������������7<IVlqtxyxunUI=:9647�
$!�������().5BENQSQNJB?5--,)(yz{��������zyyyyyyyy#09<70.#=HUaxz�}zrnaUOHGCA=��

���������������
 !#$#
� ����/�$�#����#�*�/�<�D�H�R�H�>�<�/�/�/�/�ֺкɺɺɺֺ̺ۺ������
�
��������U�Q�U�^�a�n�x�r�n�a�U�U�U�U�U�U�U�U�U�U�A�8�4�(� �&�(�.�4�A�M�O�Z�]�\�Z�Q�M�A�A������#�/�<�=�D�<�9�/�#�����������������������������������������������x¦²µ´²®¦�������y�~�������������������������������������������������$�0�7�=�>�:�0�$�����������������������������������������������������������������������������������6�4�5�6�<�B�O�S�[�h�p�h�a�[�O�B�6�6�6�6�(�'� �"�(�4�6�A�M�Y�Z�]�Z�S�M�B�A�4�(�(�������������������������������������������� �$�$� ������ʾ����������������������ʾ׾����׾�����������Ƶ�����������������������������������(�5�A�E�N�O�U�N�J�A�5�(��Y�4����5�@�M�\�f�r�������������f�Y������������������������������������������y�t�r�t�|������������������������	�����ھؾ���	���"�)�.�4�6�.�"���óñëÜ×Øàì���������������������ſ`�T�P�J�I�T�`�m�y���������������}�y�m�`�ɺº����������ɺֺ���� ��������ֺɾ������������(�4�5�A�E�I�A�7�(���6�,�*������*�6�C�O�Q�O�C�C�6�6�6�6�H�C�D�H�O�U�V�a�b�d�a�U�H�H�H�H�H�H�H�H�	���������������	��"�+�/�;�=�;�/�.�"�	�Ľ����������Ľнݽ�����ݽսнĽĽĽ��U�U�H�F�C�A�G�H�P�U�Z�[�Z�U�U�U�U�U�U�U�������~���|�������Ŀݿ����� ����ѿĿ������������	��������	�����ŠŔŏŒŔŠŭŹ������������������ŹŭŠŭūũţŧŭŶŻ��������������������Źŭ�нĽ����������������Ľнݽ������ݽ������������������������������������������a�n�x�t�n�d�a�`�\�a�a�a�a�a�a�a�a�a�a�a�A�:�5�A�N�W�Z�[�Z�N�A�A�A�A�A�A�A�A�A�A���������������ĿͿϿĿ������������������l�i�l�o�x�������������x�l�l�l�l�l�l�l�l���������ʼӼ����+�1�.�����ؼʼ������Z�T�G�H�M�M�Z�f�k�s�������������s�f�Z���������������þʾ׾ھھ׾Ҿʾ���������ĦĚĊĂĀĆčĚĦĿ����������������ĿĦ����������������������� �
����
�������T�Q�H�>�B�H�T�a�b�m�t�z�����|�z�m�a�T�T�	�����������	��"�)�.�/�0�/�.�*�"��	�%�#�4�S�l���������Ľ���ݽĽ��y�S�:�%������������������������������������������׾̾ɾ;׾����	��"�.�1�"��	�����ā�}�u�wāčĚěĦİİĦĚčāāāāāā���������&�(�-�(�����������L�I�E�>�=�@�J�Y�e�r�~���������r�e�]�Y�L�f�d�f�p�{�������������¼ļļ���������f�b�U�C�<�9�C�I�U�b�n�{ŁņŊŇņń�{�n�bŠŝŘŔŇŁŇňŔŠŭŰŵŴŭŢŠŠŠŠ�Ľ½��������ĽʽнҽнŽĽĽĽĽĽĽĽĻ����x�u�l�g�b�c�l�x���������������������������'�2�4�:�@�L�@�9�4�'����¿¦¨ª²¿������������������¿�ɺǺ����������ɺպֺ�����ֺɺɺɺɼ�������ݻ�������'�1�.�'�����û������������������ûȻûûûûûûû��6�,�)� ��)�+�6�?�B�F�O�O�O�C�B�6�6�6�6������������������������������������������������������������������������������������������� �����#����������������!���!�-�:�F�S�_�l�x�������l�_�F�:�-�!��������������������	����������������ÓÒÇÀÄÇÑÓàìñù��������ùìàÓ���������������ĿǿǿĿ����������������������������ûлٻܻ��ܻлû��������������������������ùϹֹܹݹ޹عϹƹù�����D�D�D�D�D�EEEEEED�D�D�D�D�D�D�D�D�E*EE*E-E7E>ECEPE\EbEiEtEqEmEiE\EPECE;E* D  Y ! 3 + .  ( + 9 > : 1 2 + �  3 N 9 F w K  " x 0 C - y ' D  0 9 J W d b U n 1 a  @ 0 4 Y 8 . B V 3 U . 5 J / J > H ? ` # 3 $ G ; ' 8 * ( : L i  �  ?  \    �  �    �  *  �  f  �  �  ,  �  �  �    e  M  G  �  g  �  z  l  �  \  U  �  �  �  ]  �  N    �  ,      d    0     s  �    �  \  �    �  �  �    �  �  j  S  �  �  �  4  L  �  �  �  �  k  �  5  g  �  �  �  <  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  �  w  d  S  A  .      �  �  E  �      b  z  w  p  `  O  <  %    �  �  �  6  �  �  4  �  �  �  �  �  �  �  �  �  �  �  y  R    �  �  �  M    �  �  ^  �  �  �  �    $  )  ,  <  D  :  !  �  �  �  R    �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  {  e  O  :  ,         �  �  �  �  �  �  y  q  j  c  [  R  J  B  9  /  $        )  K  j  |  �  |  k  N  )  �  �  �  e    �  e  �  �  !  :  7  0  '      �  �  �  �  �  ~  _  ?        �  �  �  {  U  2  A  8  0        �  �  �  �  f  -  �  �  �  p     �   �  �  �  �  �  �  �  �  �  �  |  c  K  2    �  �  �  �  q  H  K  I  F  D  A  ?  <  9  6  2  /  +  (          �  �  �  �  �  �  �  �  �  �  �  �  h  <    �  �  �  �  �  z  f  d  o  �  �  �  �  �  �  �  �  �  �  �  u  Z  ;    �  �    i   �  �  �  �  �  �  �  �  �  �  �  �    }  {  y  v  t  r  o  m  !  #  %  %      	  �  �  �  �  �  �  �  
  �  +  �  }  $  �  	              �  �  �  �  �  �  �  �  `    �  0  �  y  ^  C  )    �  �  �  �  �  �  x  g  V  ;     �   �   �  �    $  4  B  I  Q  [  c  `  N  /    �  �  �  *  �  C  �  #  ?  Y  k  h  K    �  �  )  �  0  �  �    !    �  s  +  y  �  �  �  o  W  8    �  �  s  /  �  �  I  �  �  S  �  �  �  �  �  �  �  �  �  �  �  g  D    �  �  g  ;    �  �    �  �  	          �  �  �  �  �  �  Y  �  m  �  S  �  >  f  a  T  ?  %    �  �  �      '  �  �  �  �  g  B    �  �  �  �  �  �  �  t  c  R  @  ,    �  �  �  �  �  �  }  7  ,  3  0  <        �  �  �  �  �  �  ~    �  �  .    .  7  I  R  X  X  S  F  0    �  �  �  �  s  K    �  �  �  �      �  �  �  �  �  �  �  �  �  �  ~  s  h  ]  R  F  ;  0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  f  (  �  �  �  �        	  	    �  �  �  �  �  >  �  �  �  w  I    ^  q    �  �  �  �  p  R  /    �  �  w  C    �  �  r  �  c  a  ^  \  X  T  O  I  C  =  7  0  )  !        �  �  �  >  /  *  =  9  /      �  �  �  V  (       �  �  a  �  >  �    ,  5  <  A  D  E  A  4      �  �  �  Q    �  �  R  $  y  �  �  �  �  �  �  �  �  �  �  Y    �  �    w  �   �  v  i  \  T  K  <  +      �  �  �  �  �  o  R  5    �  �  [  X  J  D  N  U  l  u  w  k  T  3  �  �  S  �  �  F  �    �  �  �  �  �  �    f  K  .    �  �  �  �  c  =    �  b  '        �  �  �  �  �  �  �  �    |  {  �  �  �  �  �  :  6  2  -  )  %  !            �  �  �  �  �  �  �  �  s  _  K  7  !    �  �  �  �  �  h  J  -    �  �  o  5   �  d  R  @  ,    �  �  �  �  �  �  }  [  :    �  �  �  �  ^  �    �  �  �  �  �  �  o  R  )  �  �    �  _  �    ]   %  D  9  *      �  �  �  �  \  )  �  �  �  V  !  �  �  �  �  �  �  �  �  �  �  u  a  L  7  !    �  �  �  �  �  z  `  F  y  �  �  �  v  j  R  8        �  �  �  p  =    �  �  ]  �  e  G  +    �  �  �  �  �  �  �  �  �  �  �  y  B  �  �    �  �  �  �  �  �  �  �  |  s  k  \  G  -    �  �  �  �  �  �  �  �          �  �  �  �  x  K    �  ^  �  {  �  9  H  `  O  =  !  �  �  �  F  �  �  [  7  �  �  �  �  �  4  �  �  �  �  �  �  �  �  �  v  Z  :    �  �  f  -  �  �  �  P  `  i  j  w  {  v  o  d  V  B  %  �  �  }  .  �  l  �   �  �  �    {  p  c  T  B  -    �  �  �  {  E    �  N  �  U  ,  n  �  �  �  �  �  �  x  m  b  V  K  @  7  /      �  �    s  g  N  3    �  �  �  �  u  O  %  �  �  �  �  b  2  �  H  �  �  �  |  T  (  �  �  S  �  �  )  �  R  �  P  �  �  @  �  �  �  �  �  �  }  g  O  0    �  �  _     �  �  ^  �  �  �  �  �  �  �  �  �  �  �  m  K  '     �  �  �  z  U  .    Y  T  N  I  C  =  8  2  -  '       �   �   �   �   �   �   �   �  �  �  �  �  �  d  E  /  '  	  �  �  0  �  t    �  ,      Z  J  ;  +         �  �  �  �  �  �  �  z  ^  E  <  3  *  �  �  �  t  i  l  �  �  �  �    m  S  !  �  �  8  �  s  �  �  �  �  �  �  �  �  �  l  U  =  &    �  �  �  �  m  >    �  �  �  r  V  6    �  �  �  e  E  -    �  �  �  W  	  �    #  /  :  F  R  ^  e  i  n  r  v  z    �  �  �  �  �  �  �  �  �  �  �  o  P  /    �  �  �  �  �  x  [  <    �  i  :  -  !          �  �  �  �  �  o  P  /    �  �  &  y  �  �  �  �  �  �  �  �  �  �  �  |  o  a  Q  A  .    �  �  �  �  �  �  �  �  ~  v  o  i  c  ]  R  C  4  &    �  �  �  �  �  �  �  �    `  >    �  �  �  b  "  �  �  7  �  D   �  h  n  �  �  �  �  �  �  �  �  �  �  �  [  *  �  �  �  o  �              �  �  �  �  �  �  �  g  9    �  f    �  �  �  �  �  x  W  8    �  �  �  g  8    �  �  �  ^  8    �  �  �  �  �  �  �  �  �  n  V  <    �  �  S  	  �  r  ,  g  r  [  ?    �  �  �  g  6    �  �  H  �  �  _    �  �  	�  
  
!  	�  	�  	o  	0  �  �  n  +  �  �  \    �  y  ;  �  �  �  �  �  �  |  G    �  �  o  1  �  �  x  ?    �  �  �  M