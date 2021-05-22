CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�
=p��
        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       P�@0        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�9X        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?c�
=p�   max       @F��\)     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @A�          @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�@            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �I�   max       ;o        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�O�   max       B4�         :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�P�   max       B4��        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       > �   max       B�o        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >A��   max       B�M        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          [        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       Pm�        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ح��U�   max       ?�������        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <���        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?c�
=p�   max       @F�ffffg     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @A�(�\     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @�@            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Am   max         Am        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u��"    max       ?�������     �  ^�                              Z         1         3                                 $               &         +               -   0      	      #   6               	   
                           !         C                          O��rO�N�/7OD�zN"��Nm��O��O�)N�P�@0Og��O�ϾO�Z�N���O_�O���N���Nz��NPuOL�O��}Os3�O��`N�DwO�F-ND={Pe�O.�cN��|N���O5�O��HOn��Nރ�Oy�LN&�PN��Nj�O�k0O��aP:�N�&LNQ&Nu|�OLx�P5�OG�*N&I�NK�xOh{�O��O \�N���O��,ONN���Nc�N��)N*�FN��|P-�N]�EOlU
Pm�O�� Oa�Nn��O�QWO��N��OO fGO8��<�9X<49X<49X<o;��
;�o%   %   ��o�D����o���
�49X�D���D���T���e`B��o���㼛�㼛�㼣�
��1�ě��ě�����������/��/��h�����o�\)�\)��P��P�'0 Ž0 Ž49X�49X�49X�<j�<j�@��@��@��Y��Y��Y��]/�]/�e`B�ixսixս}󶽅������+��7L��C���\)��hs��hs���
���T�� Ž� Ž�-��-���mpx�������������zvpmRYYXYXacz�������znaR�������������������������������������
 ��������|�����������||||||||�������������������������������������rt�����ttrrrrrrrrrrr����+6<:,���������������������������5BNThq������tg^KF=)5=BO[g�������{thOB>:=��������������������$6?COTTVWXUOC6*������������������������������������������������������������HO[hmhe[OFHHHHHHHHHHt�����������tqokkjmt��������������������#/<BHNMHB<7/#)5DN[js[NG5)GHTWaaccaTHECFGGGGGGlsz�������������zmilS[httutkh_[USSSSSSSS��#8U{�����{h0#
����������������������lnt{��������{rnkllll
#)/1/# 
	����������������������������!&+$���zz�����������������z�����������}������������������}nt������xtpjnnnnnnnn;BOP[_[OB<;;;;;;;;;;rt���������trrrrrrrr��������������������HUaz�������znaUKHBAH'B_ro`^_[N)		
����� 
!�����)6:=6)}�����������|���}}}}�����������������������)46:=?;)���������������������������������������������hmtz{�zumiehhhhhhhh���������������������������������������NNU[git�����~tg[SNNN��������������������#0;IKUW_a_UP<0&$)������������������������������������������������������������;<IU_bkdbbUIH@<:;;;;��������������������MOR[hkjhhfc[XTONMMMMal���������������tdaehtv}���thfeeeeeeeee��������������������BNj������������ygC:B��������������������GHKUajnwz}~~zaUJGGG��������abnz�����������zsndaOORU[hnqtuywoh[VOKKO&0<GIJIHF<;10,&&&&&&05;<CIPTVURKI<40-,-0���������������������������������������*�8�C�G�C�(��������������s�i�g�[�_�s�������������������#� ��� �#�/�;�;�6�;�/�#�#�#�#�#�#�#�#�A�7�;�A�M�O�Z�s���������������s�Z�M�A����׾Ҿ׾۾���������������������������������������������������������������������
��#�/�<�@�<�6�/�#��
������¿²¦¥¦²¿���������������ؽ����������������������������������������g�W�A�"�(�*�5�Z�g�������������������s�g�`�T�R�K�I�T�Y�`�m�y���������������x�m�`�y�o�o�y�����Ŀݿ�����������������y�x�l�Z�]�l�����������û����ƻлȻ������x�U�M�H�@�<�4�6�<�H�T�U�[�[�\�U�U�U�U�U�U�	���߾׾Ѿ̾����	��"�'�-�.�-�*�"��	�����'�@�M�U�f�r������f�Y�M�@�'���x�x�l�j�_�\�S�R�S�S�_�h�l�n�x�|�{�{�y�x���������������������������������������������z�~����������������������������������پ۾����	�"�.�;�A�J�G�;�.�"��	�����s�f�k�n�f�c�d�s���������¾ʾҾʾ�����/�#���#�/�@�H�U�[�a�j�n�q�n�[�U�H�<�/�z�m�a�]�U�U�a�l�v�z�������������������z�������������	������	���������������A�5�.������������(�E�T�Z�V�M�S�N�A�Ŀ����ĿͿѿѿݿ�ݿܿѿĿĿĿĿĿĿĿ����������v�g�O�=�5�g��������������������������	��"�/�;�H�T�X�W�T�H�;�/�"��������������������ʼ̼μϼʼü����������O�G�B�6�2�1�6�>�B�O�X�[�h�t�u�y�t�h�[�O��������������������$�)�0�-�)�����r�Y�M�'���������4�M�Y�f�{�������������ĿĹĻĿ�����������
��������׾ԾʾǾʾ̾׾������������׾׾׾׿Ŀ����������Ŀݿ�����������ݿѿɿ��6�5�2�6�C�C�C�J�O�Q�O�C�6�6�6�6�6�6�6�6�a�^�a�a�k�n�s�z�w�n�a�a�a�a�a�a�a�a�a�a����ûùìù�����������������������������`�V�P�I�F�>�I�V�o�{ǈǋǎǌǌǈǄ�{�o�`���������������ʾ���	��������׾ʾ�ŠŤŭž����������*�6�?�;�3�*����ŹŠÓËÇ�z�w�z�|�z�y�x�zÇÓ×ÝàáàÚÓìãàåìù��ÿùùììììììììììŹųŭŨŭŹ��������������������ŹŹŹŹ����ŹŭŬŢŠŧŭŹ��������������������àÓÇ�n�`�P�a�zàì����������������à�m�`�Z�T�Y�`�m�����������������������y�m�ѿȿĿ��Ŀѿݿ����ݿѿѿѿѿѿѿѿ���Ƽ�����������������������������������������������������$�/�3�2�0�$��������a�a�g�p�z�������������������������z�m�a���������ܾݾ�����	����	�������׾;ʾȾɾʾо׾�������׾׾׾׾׾׻лŻʻû����������û����%�����ܻк����������������ɺֺ�����ֺܺɺ����r�h�e�_�e�p�r�~���������������~�r�r�r�r�����	�������������������������������-�,�+�,�-�0�:�F�S�S�T�V�S�F�A�:�-�-�-�-�ּʼռּ��������������ּּּּּ�������������������������������������������ĿİĦĔĄĀĞĳ�����������
����������������	���� ���	�����������������O�K�B�H�P�_�h�tčĚĦĩĤĚčā�h�c�[�O��½¦¦������/�a�g�a�H�/��
���Ժ����'�I�e�r���������������~�e�@�'����������������������ùϹܹ���Ϲù��������ݽսݽ���������������������a�T�D�?�B�K�T�a�m�z�����������������m�a���������������������ùϹܹ��ܹչϹù������������������ĽǽĽ������������������������ݽԽ����(�4�6�;�4�/�(���������!�!�.�G�S�Y�\�`�f�`�Y�S�:�6�!� N 3 X ` T l U V 6 7 ( � N u < = l N B c > ' 7 5 F > l G = ` : e > / R J [ h F - : W V J ! _ @ Z ^ , � ? i T " S ; C l ] Y R _ P x 4 g 5 8 ? P F  h  2  �  �  [  �  z  �  -  ^  �     e     �  �  ?  �  s  �  �  �  �  �  l  o  \  �  �     �        �  .  V  d  d  W  �  ?  �  �  y  �    �  8  �  �  �  4  	  �  :  �  ^  �  w  �  &  |  +  j  �  �  �    C  �  �  ���o�o�ě���o;o:�o���ͼ�j���
��vɼ�C���/�q����������o��/��9X��9X���'@��D����/�T����`B�y�#�,1�C���w�,1��O߽aG��L�ͽ��-�#�
�H�9�@��}󶽲-��^5�aG��Y��L�ͽ��
�����%�H�9�e`B��\)�y�#��%�u��hs���T��������hs��C����
���ͽ�t�����I�����񪽰 Ž�`B���Ƨ�������B �"B6�B�FB�B%B,�B\�B1B��BFB�JB	DEBB EB0d�B �bB"�B4� B"�B�;B ��B�Bq�A�O�B�B�PB&��BLPB)QB_�B1�B_�B ЪBm�Bb�B
�B��B
]�B.B��B<�B�B,�B
�|B��B�B*��B*��A���B��B�hB	d�B��B&��B"�B"ͶBA�B'�B-53BY�B�xB/)B�B
��B�BHLB4*B��BB&lB&{�B!ZB ��BC�B@B?6BKwB	�BC�B@�B�(B>B�ZB	��Bj�B ��B0D�B � B">xB4��B9#BͤB @,B��B@A�P�B@#B�:B&�JB?�B(��BF[B�[B6AB �+B=B}�B	�B��B
�jBm=BA�B��B@B5�B
�.B��B��B*��B*qQA���B�BHB	hB��B&��B"��B"��B?�B&�VB-<�B?�B
�AB6�BD=B	�{BB�B<�B�XB��BP�B%�B&B�B��A��%A�W1A�	PAB�AU�qA�D}A�XcA�\!A ��A� Ak��Ax*m@�CAč�AZh�@�`R@�/�ALl�Ao�hA\s�AH�AĄA���A�=A�_yAz>�A��,A���@��	A��lA�>@�ZA��AUgZA}� B ��A�6�A�7�B�oAS[�A� KA��$A���A�1\A��7A��uAm��A|�Bx�BԼA�AX!>AT
�@��@3F@B4@}��A{BA�`�A���A��Aܫ?A���?�f�> �A/B~A��
>6��A"�A2��A��A���A���A�}�AC AT�A�_[A�jMA���A A��AkiA~�@��AĉhA[d@���@��ALy�Ao,NA]��AGMFA��A��lA�ͫA��hAzOA�XjA��Y@�
~AٰjA��@�	A��AT��A~=B �}A��A͋�B�MAT��A�AʀA�}�A��A�%�A�y�AkNA|t�BU�B	�A���AYeAS�@��@3�L@�Bϝ@|�A�BA�yRA��A���A�~dA�~U@�'>A��A0�!A��>B��A!PA5�A��                              [         1         4                                 $               '         +               -   0      
      #   7               	   
            	               "         C   !                                                    9      -   )         #               !      !      !      9               )                        !   +               +                        !                     3         9   %                                                         -   )                              !            9               )                        !   %                                       !                     3         9   %                     O�7�O�NQ�(N�(!N"��Nm��O��O N�O��aO=��O䚄O�Z�NF OIz�O@�"N]NH!INPuOL�O�-�N���O��`N�DwO.��ND={Pe�N��rN��|N���O5�O��HOn��NƓ$OX8N&�PN��Nj�OYϭO��aPm�N�&LNQ&Nu|�O��O�WO�#N&I�NK�xOS�UO��O \�N���O�UN�W5N���Nc�N��)N*�FN��|P-�N]�EOlU
Pm�O�� Oa�Nn��O�QWN�Y�N��OO��O8��  Y  �  '  �  �    �  �  �  _    j  !  %  +    r  9  7  E  �  �  Q  �  J  m  >  �  �  :  m  <  t  {  �  o  �  �  H  !  �  d  �  a  F  �  �  �  >  k  H  :  �  �  ^  �  �  �  <    �  �    �    �  �  �  �  �  j  �<���<49X<t�;D��;��
;�o%   ��o��o�]/�ě��ě��49X��t��T����󶼃o��C����㼛�㼬1��󶼬1�ě���������������/��h�����o�t��@���P��P�'<j�0 ŽY��49X�49X�<j�Y���o�H�9�@��Y��]/�Y��]/�]/�ixսy�#�ixս}󶽅������+��7L��C���\)��hs��hs���
���T�� Ž\��-��E����r{�������������zxsprRYYXYXacz�������znaR������������������������	 �������������
 ��������|�����������||||||||����������������������������� ������rt�����ttrrrrrrrrrrr�����

��������������������������,BNgp������tg_ZWLG4,=BO[g�������{thOB>:=��������������������*-6@CORSVVSOC6*������������������������������������������������������������HO[hmhe[OFHHHHHHHHHHt�����������tqokkjmt��������������������"#/17<>?<3/#""""")5DN[js[NG5)GHTWaaccaTHECFGGGGGG��������������������S[httutkh_[USSSSSSSS��#8U{�����{h0#
����������������������lnt{��������{rnkllll
#)/1/# 
	����������������������������!&+$���zz�����������������z�� 
����������������������������nt������xtpjnnnnnnnn;BOP[_[OB<;;;;;;;;;;rt���������trrrrrrrr��������������������HUaz�������znaUKHBAH)[`dZVWNH5)		����� 
!�����)6:=6)}�����������|���}}}}���������������������)+./33-)���������������������������������������������hmtz{�zumiehhhhhhhh���������������������������������������NNU[git�����~tg[SNNN��������������������#0<IUZ`^[UO<0'%'+������������������������������������������������������������;<IU_bkdbbUIH@<:;;;;��������������������MOR[hkjhhfc[XTONMMMMal���������������tdaehtv}���thfeeeeeeeee��������������������BNj������������ygC:B��������������������GHKUajnwz}~~zaUJGGG��������abnz�����������zsndaNOTZ[hiotttphhh[PONN&0<GIJIHF<;10,&&&&&&07<<HINSUTQJI<50.-.0����������������������������������������*�2�@�=�"����������������s�i�g�[�_�s�������������������#�"���#�%�/�8�9�4�/�+�#�#�#�#�#�#�#�#�Z�V�Z�Z�f�s�����������s�f�Z�Z�Z�Z�Z�Z����׾Ҿ׾۾���������������������������������������������������������������������
��#�/�<�@�<�6�/�#��
������¿²¦¥¦±²´¿�����������ؽ����������������������������������������s�g�Z�N�M�R�Z�e�s���������������������s�`�T�M�Q�T�^�`�m�y���������������y�s�m�`���r�r�����Ŀݿ����������ݿ��������x�l�Z�]�l�����������û����ƻлȻ������x�H�G�?�<�:�<�H�O�U�W�V�U�H�H�H�H�H�H�H�H��	����׾Ӿؾ��	��"�%�+�-�,�'�"�����'�4�@�M�Y�f�n�r�t�l�f�Y�M�@�4�'��l�l�a�_�X�_�a�l�u�x�{�y�x�m�l�l�l�l�l�l���������������������������������������������z�~����������������������������������پ۾����	�"�.�;�A�J�G�;�.�"��	�����s�i�n�p�i�g�l�s�������������ž�������/�.�,�/�<�D�H�U�X�a�e�b�a�U�H�<�/�/�/�/�z�m�a�]�U�U�a�l�v�z�������������������z�������������	������	���������������(�#����������(�5�8�A�L�N�H�A�5�(�Ŀ����ĿͿѿѿݿ�ݿܿѿĿĿĿĿĿĿĿ����������v�g�O�=�5�g���������������������"������"�+�/�0�;�H�T�T�T�H�F�;�/�"�������������������ʼ̼μϼʼü����������O�G�B�6�2�1�6�>�B�O�X�[�h�t�u�y�t�h�[�O��������������������$�)�0�-�)�����r�Y�M�'���������4�M�Y�f�{�������������ĿĹĻĿ�����������
��������׾־ʾȾʾξ׾���� �������׾׾׾׿Ŀ����Ŀɿѿۿݿ�����������ݿѿ��6�5�2�6�C�C�C�J�O�Q�O�C�6�6�6�6�6�6�6�6�a�^�a�a�k�n�s�z�w�n�a�a�a�a�a�a�a�a�a�a����ûùìù�����������������������������o�j�b�V�Q�M�L�V�b�o�{�}ǈǉǋǉǉǈ�{�o���������������ʾ���	��������׾ʾ�ŸŴŸ�����������1�6�8�5�*�������ŸÓËÇ�z�w�z�|�z�y�x�zÇÓ×ÝàáàÚÓìãàåìù��ÿùùììììììììììŹųŭŨŭŹ��������������������ŹŹŹŹ����ŹŭŧŦŭŮŹ����������������������ÓÇ��z�y�}ÇÓàù����������������àÓ�y�n�m�`�\�V�\�`�m���������������������y�ѿȿĿ��Ŀѿݿ����ݿѿѿѿѿѿѿѿ���Ƽ�����������������������������������������������������$�.�0�2�2�0�$�������a�a�g�p�z�������������������������z�m�a���������ܾݾ�����	����	�������׾;ʾȾɾʾо׾�������׾׾׾׾׾׻лǻ̻Ż������û�����$�������ܻк����������������ɺֺܺ����غֺɺ����r�h�e�_�e�p�r�~���������������~�r�r�r�r�����	�������������������������������-�,�+�,�-�0�:�F�S�S�T�V�S�F�A�:�-�-�-�-�ּʼռּ��������������ּּּּּ�������������������������������������������ĿİĦĔĄĀĞĳ�����������
����������������	���� ���	�����������������O�K�B�H�P�_�h�tčĚĦĩĤĚčā�h�c�[�O��½¦¦������/�a�g�a�H�/��
���Ժ����'�I�e�r���������������~�e�@�'����������������������ùϹܹ���Ϲù��������ݽսݽ���������������������a�T�D�?�B�K�T�a�m�z�����������������m�a�ù��������������¹ùùϹܹܹܹڹϹȹùý����������������ĽǽĽ�����������������������ݽݽ����(�4�5�:�4�-�(��������!�!�.�G�S�Y�\�`�f�`�Y�S�:�6�!� H 3 _ E T l U L 6  $ � N f 8 A ] 5 B c 4 ' 7 5 F > l K = ` : e > 1 @ J [ h < - 6 W V J * A ) Z ^ % � ? i O ' S ; C l ] Y R _ P x 4 g 5 ( ? N F    2  �  �  [  �  z  V  -  %  �  �  e  D  �  �  �  M  s  �  j  �  �  �  u  o  \    �     �        �  M  V  d  d  �  �  [  �  �  y  1  y  Y  8  �  �  �  4  	  s  �  �  ^  �  w  �  &  |  +  j  �  �  �    �  �  e  �  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  M  V  Y  T  E  7  ,      �  �  �  ~  E    �  h    �  �  �  �  �  �  �  �  �  �  �  l  A    �  �  �  �  {  _  =    �  �    B  d  �  �  �  �  �  �  �  �  u  Q  *    �  �  w  �  �  �  �  �  �  �  �  �  �  �  �  y  b  F    �  �  _    �  �  �  �  �  �  �  �  �  �  �  �  �    |  x  u  r  n  k      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  a  ?    �  �  �  E  �  �  6  �  �  )  `  ^  �  �  �  k  8    �  �  ~  W  D    �  }    �  M  �  i  �  �  �  �  �  �  �  �  �  �  �  y  o  e  [  Q  F  :  /  #    L  �    7  f  �  �    #  =  V  ^  A  �  �    {  �  =  �  
            �  �  �  �  �  k  =    �  �  l  L     �  g  j  f  X  D  +    �  �  �  �  o  4      �  �  N  �  {  !    �  �  �  �  �  �  �  �  g  1  �  �  [  �  �  1  �  M                   $  !         �  �  @  �  �  F  �     +  &        �  �  �  �  �  �  �  q  W  6  �  �  9   �    R  �  �  �  �        �  �    &  �  I  �  R  �  �    ,  C  [  s  ~    y  m  _  P  ;  *      �  �  �  s    �  4  6  7  9  1  (        �  �  �  �  �  �  �  �  z  b  K  7  4  1  .  *  '  $  !               �   �   �   �   �   �  E  6  '      �  �  �  �  �  Z  .  �  �  �  U  !   �   �   �  �  �  �  �  �  �  �  j  L  -    �  �  �  Z  5    �  U   �    .  F  ]  q  �  �  �  �  �  �  z  ^  7  	  �  �    _  �  Q  6    �  �  �  �  g  0  �  �  �  �  �  �  \    �  -  ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  m  c  Y  O  E  �    '  ;  F  J  G  ?  6  &    �  �  |  1  �  z    �  8  m  k  i  g  f  d  b  `  ^  \  Z  X  V  S  Q  O  M  K  H  F  >  6    �  �  �  x  M  5    �  �  p  .  �  �  S  �  �  L  �  �  �  �  �  �  �  �  �  �  �  �  z  Y  *  �  v    �    �  �  �  u  d  S  >  %    �  �  �  �  �  �  [  5     �   �  :  4  .  '           �  �  �  �  �  �  �  v  _  E    �  m  d  `  _  [  V  Q  L  F  :  )       �  �  �  �  �  Z  #  <  1       �  �  �  j  �  �  v  8  �  �  B  �  �  >    �  t  m  X  A  *    �  �  �  q  >  	  �  �  D  �  �    s   �  n  y  {  v  o  h  ^  S  C  -    �  �  o  5  �  �  y  *  �  [  �  �  �  �  �  �  �  �  �  �  [  &  �  �    �    �    o  g  _  W  O  G  @  9  4  .  )  #                  �  �  �  �  �  �  �  o  P  ,  �    �  �  [    �  �  f  (  �  m  W  B  %    �  �  �  �  g  /  �  �  �  �  w  Z  =     :  9  7  H  ;  ,      �  �  �  �  |  U  )  �  �  y    o  !    �  �  �  �  �  �  �  �  Y  &  �  �  p    �  7  �  X  �  �  �  �  �  �  �  �  �  �  l  <  '  &    �  �    k  8  d  `  Z  P  H  B  .      �  �  �  �  �  r  6  �  �     �  �  �  �  �  �  �  �  �  �  v  g  b  j  j  W  D  2    �  �  a  ^  [  W  T  O  B  5  (      �  �  �  �  �  �  �  k  U  �    2  @  F  A  1    
  �  �  �  O  �  �  1  �  �  S  �  D  ~  �  �  �  �  �  �  k    �  X  �  j  �  p  �  �  �  �    �  �  �  �  �  �  |  d  G  &  �  �  �  j  1  �  �  }  3  �  �  �  �  z  t  n  i  c  ]  Y  W  U  S  R  P  N  L  K  I  >  5  ,  $      
    �  �  �  �  �  �  �  �  �  �  �  �  Q  h  g  `  T  E  3     
  �  �  �  �  V  !  �  �  a    �  H  8  )      �  �  �  �  �  {  U  .     �  �  �  �  _  ,  :  /  $    
  �  �  �  �  �  �  �  |  c  H  )  �  �  �  o  �  �  �  �  �  �  �  f  J  .      �  �  �  �  s  F     �  �  �  �  �  �  u  f  Y  N  C  4       �  �  s  >    �  '  Q  W  [  ^  ]  T  F  5  !    �  �  �  T    �  �  9  �  �  �  �  �  �  �  �  r  b  Q  A  1  "    	  �  �  �  �  �  v  �  �  �  Z  3  
  �  �    V  ,    �  �  r  :  �  �  v  /  �  �  �  �  �  �  �  t  Z  A  '    �  �  �  �  �  x  `  G  <  6  1  +  %           �  �  �  �  �  �  �  �  �  q  `    �  �  �    a  D  &    �  �  z  H    �  �  a  "  �  �  �  �  t  O  9    �  �  �  �  u  X  E  ]  T  "  '  �  L  �  �  ~  q  d  W  J  <  -      �  �  �  �  �  �  g  K  /               �  �  �  �  x  Z  ;  "    �  �  �        �  �  �  g    �  �  I  |  �  �  i  ,  �  Q  �  V  �  3  =  �    �  �  �    R    �  k    �  G    �  �  n     �  ?  �  �  �  k  Q  3    �  �  �  v  F    �  �  d  '  �  g  �     �  �  �  �  �  w  m  b  X  L  A  6  *        %  M  v  �  �  �  �  �  �  �  t  P    �  �  T    �  y  7  �  �  �  b  �  �  �  �  �  �  �  �  �  �  �  �  �  v  C    �    �  H  �  �  �  �  d  >    �  �  �  [  #    +      �  �  �  �  \  g  b  S  D  =  0  
  �  �  d  '  �  �  L  �  �      �  �  c  ~  d  H  ,    �  �  �  ~  e  K  0  
  �  �  5  �  P