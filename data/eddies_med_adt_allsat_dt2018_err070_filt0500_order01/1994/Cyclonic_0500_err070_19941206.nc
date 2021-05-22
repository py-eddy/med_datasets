CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�M����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N ��   max       Pm�k      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <o      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @Fu\(�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vzz�G�     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @�`          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       ��o      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0w@      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0C_      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�f%   max       C��I      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�   max       C���      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N ��   max       Pm�k      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?���Z�      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <o      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @F=p��
>     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vzz�G�     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P@           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @�`          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @   max         @      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�_��Ft   max       ?�@N���     `  U�   !         
               
      �         *             	               	         >                     #   7               	               E               ,                                       PkڡNk�jN�jN�M�Oy�Na=N.ӔP[yOW|�N7�$PY:�O�%�O�RHP��N��O�G7N��&Nv�;Nd�WO���N ��N���O��O��0N2�uPm�kN�O/P8N��RO�aNŔ�Op��O��PPA�dN��~N�z�O �N��O��O9+�N��ZO]&N��dOG��O�U�N�O�{NK�-OG�O-4N���N�O_�(NrIO<j�O�COD��N�M�O�nNq�N=�O
�<o;��
�o�o��o��o�ě���`B�o�t��t��#�
�D���D���T���T���u�u��o��C���t����㼛�㼬1������`B��h�o�+�t���P������w�8Q�D���H�9�H�9�L�ͽY��]/�]/�]/�m�h�y�#�����������+��+��O߽�t�������㽝�-���-��^5�\��������������HUnz����������UOH=?H�������������	  ������oty���������tsomoooo35:BDN[dgkjg`[NB6533��������������������45BNQRNB754444444444���������������������������������������������������)6>=76���������rz�������������xsnr����

��������)./)*6Od�����ll[O6+)��������������������*6CO\h���{j\6�������������������������

����������������

�����������7?BN[hu����tg[NB5+��������������������[[[gkplkljg[YWONPU[[���������
�������������������������������������������������#Ibn{���{<
��36BCJKLIB630/-333333mz���������������zjm��������������������#/8<GGHUZUJH</#`amz�����zrmka\Z````����������������������������
��������FPaz����������aUKILF����������������������������������������~���������������||~�����������������������
#&008;30+#
��IO[hqtxuqrsmhc[RHGEI=BHN[^][ZVNB;<======djnot�����������tg`d{���������������{{{{��
 #&('$
���������������������������������������������������������������������������<CFGGE@<4/#  #/<�����


��������@BKNR[\\[Z[[[NJGB@@@��������������������#0<IMURQLI<0#[[hhjlkha[[Z[[[[[[[[
#/<?<<<4/,#
GJXdnz��������zaUH?G�������nt��������vtnnnnnnnn&)+-5;BKNTYZNMB5)(&&MNS[\gt���ttgb[NMMMM)/5<?BD</'))))))))))+/9<HUV[ZUQH</,'%'++�W�L�_�c�c�o�����������������������s�g�W������|������������������������������������������Ŀ̿ѿݿ�����ݿٿѿĿ����/�&�#��#�/�6�<�H�U�Z�^�V�U�H�<�/�/�/�/�����������������������������������������y�w�m�w�y�������������������y�y�y�y�y�y�����������	���	��������������������������q�Y�J�@�:�G�T�m�����Ŀѿ����ݿĿ���������"�;�G�T�`�j�`�T�G�;�.�"��	����������� ��������������*�"���
��4�M�r���������z�r�f�Y�4�'�*�U�P�Q�T�U�I�U�b�n�{�~�yŉŎŐŇ�{�n�b�UŠŢŞšŬŹ��������������������źŹŭŠ�����x�_�S�8�B�_�k�x���������ûһӻ̻�����������
�����
���������������������"�	����ؾѾѾھ�޾���	�"�.�9�9�4�.�"�ѿʿĿ����������Ŀѿڿݿ���������ݿѹ��ܹϹƹĹϹܹ���������������Ŀ����������ĿѿҿݿݿݿֿѿĿĿĿĿĿĿ���ƾ��ľ׾���	��$�,�+�'�&� �%�"������������������������������������������y�y�y������������ĿοȿĿ����������y�yìçÛÓÇ�z�w�v�zÃÇÓ×àìòûùöì����������������)�2�5�L�O�F�5�)�����|�s�r�s�s�����������������������������������������}�k�Q�G�H�O�s��������������`�V�\�`�m�y�������������y�m�`�`�`�`�`�`ĸıĹĸĿĿĵĿ������#�)� ���������ĸ��������(�2�4�9�?�4�(�������H�=�<�.�'�/�3�<�H�K�K�U�W�a�b�d�f�c�U�H�������"�'�/�9�;�<�;�4�/�"�����L�J�G�K�L�Y�e�r�~�������������~�r�e�Y�L�������������ɺ���!�-�<�>�-�!���ֺ��M�A�>�B�M�Z�s��������ʾ������׾��s�M�g�`�Z�W�T�Z�g�s�y���������s�g�g�g�g�g�g�����������������������¾Ⱦ�������������ƳƨƧƣƧƳ��������������������������Ƴ�������������$�*�(�$������������������ܻɻû����ûллܻ�������������ù����������ùϹܹ�����������ܹϹ�����������������'�(��������������ŹŭœőŌŔŠŭŹ���������������߿Ŀ¿����Ŀƿѿܿݿ޿�޿ݿӿѿǿĿĿĿ�E
ED�D�D�D�EEEEE*E7ECEPEREQECE7E*E
�������������������ɺԺ��!�C�6���ֺ������������
�
���
��������������������"����	�������������
��%�-�7�?�@�<�"�������"�#�*�+�"���������D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxD|�������$�0�=�I�V�V�U�I�=�2�0�$�!������������������	�
������
�������Y�W�Y�^�n�r���������ʼμ�����������f�Y��������������(�4�:�G�K�<�4�!�������������������������������������������������������������������������������������z�m�T�H�B�A�C�H�Q�T�a�m�o�u����������z��������������
��)�6�B�L�M�B�:�)����à×ÚÕÚàìôõñìéààààààààÇ��z�n�b�a�`�a�b�n�zÇÑÓàááàÓÇ�U�R�H�=�<�8�9�;�<�<�H�I�Q�U�Z�_�U�U�U�UE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF!F$F'F+F)F$FFE�E�E� 4 J G 8 2 4 . Z n I T C C E ? R p U ) J j e _ ; s U L W 7 > $  M d = ] D 0 Q 5 F g � K l z y [ U ^ T � 2 j * \ � c - z ) )  �  �    �  4  u  R  �    X  �  :  2  �  :  |  0  �  p  �  F  T  q  %  r  �  �  �  �  k  �  �  �  �  �  �  z  �  F  �  �      �  �  e  �  �  �  �  �  �  �  T  �  �  .  �  V  �  J  F���ͺ�o�e`B�D���e`B�u�o�t���t��T��������o�]/����49X��j��j�����㼬1���ͼ�/�0 ż�h��E��\)�}�49X�]/�0 Ž�o��t�����T���]/�u�u�q������u���P�y�#�%���罋C����罕���;d��-���
��1��vɽ��T��vɽ�����S�����������G����mBR�B]tBc�B��BNB)�_B��B*�B\mB��B]GB �B/�B�B!�B0w@B�BJ�B=4B<B!K[B�B�B��BEB&l�BwB�B�GB�A��B!s�B"�@B��B��B ̨B �=BN#B$��Bl�B7�B
G*B}1B��B�B2BQ�B|�B-�BT�BN�B+Z^B%� B��B�*B-�BZyB
n�B�VB	4)B��B�{B>UB?�BB�B��B8�B)��B�B*�6B@B�BD1B H�B��B�jB ��B0C_B�BC�B?�B	,�B!IAB��BCWB?�B@KB&��B�B<�B��B�.A���B!��B"�BD�B��B �oB ��B�dB%<FB@	B8�B
- B=B�KB�LB
��B@B�&B=�BB&B@	B+@yB&;�B�B��B�BCVB
�yBE�B�BT�BL�A��AH6Azq�A���A�J!A9�A��dArC�A`S�@��@�	A�{�A��=@��sA���AZ��A{Z�>���AyʆAYf�@�P�AsM�A�KWA�EGA���A�Q�Al�A��A6%A�êA��	?�0c@Q!�AJ-�A�+MAKϼBB�B��@��>�f%A�0�A�\�A{JsC��J@A�'A�o�A��A��C��HB
�CA��h@�HFA4�A���A�_�A�Y�A�v(A�˰A�@A���C�^�C��IA�vBAGhAz��A�G!A�oA�$A�SAwiA`�o@�W�@���A���A�h@���A�AZ�/A{��>�m�Az�AY��@�Ar��A�w�A���A�|CA�gPAk<�A���A5+AĒnA��%?��@Tx�AM�A�]�AN��B�B	8T@��>�A���A�wA{	?C��@K��A��A��A���C���B
HNA���@�ٍA7
>A�}A�|BA���Aӓ!A�t�A�}�AĄ8C�X�C���   "                              �         +          	   	               	         >                     #   8               
               E            	   ,                                          3                     1         1         /      %            %                  7      +               )   /                                 +      %                                                3                     1                  )                  %                  7                     '   +                                 +      %                                             PkڡNk�jN�AN�M�N� Na=N.ӔP[yOW|�N7�$O���O
�oOJ��P��N��O`!�N��&Nv�;Nd�WO���N ��N���O��O��0N2�uPm�kN�O/O���N��RN\_�NŔ�OU��O쳀P)N��~N�z�O �N��:O��O;�N��ZOH9N��dO�GO�U�N�O�{NK�-OG�N��xN���N��O_�(NrIO<j�O� >OD��N�M�O�nNq�N=�O
�  �  �  g  �  k  �  �  T  [    �  �  ^  �  �  �  �  �  �  w  �  �  �    �  �  e  h  �  �  b  4  �  9  �  �  O  �  �  �  U  �  a  O  �  �  �  |  
I  Z  T  0  X    U  
  ,  �  !  �  m  �<o;��
�D���o�ě���o�ě���`B�o�t����w��o�u��o�T����j�u�u��o��C���t����㼛�㼬1������`B��h��w�+�49X��P�#�
�#�
�49X�8Q�D���H�9�L�ͽL�ͽm�h�]/�m�h�]/��+�y�#�����������+��t���O߽��������㽝�-������^5�\��������������HUnz����������UOH=?H���������������������oty���������tsomoooo6BCNN[agihg][NB96666��������������������45BNQRNB754444444444�����������������������������������������������������!%$������uz�������������|zxuu����
�������6BOh����tjj[O6,22,-6��������������������'*6COS[`_ZOC6*&�������������������������

����������������

�����������7?BN[hu����tg[NB5+��������������������[[[gkplkljg[YWONPU[[���������
�������������������������������������������������#Ibn{���{<
��36BCJKLIB630/-333333����������������������������������������!#/2;<=</#!!!!!!!!`amz�����zrmka\Z````������������������������������������JUaz��������aUMKKNOJ����������������������������������������~���������������||~�����������������������
#&008;30+#
��LOT[hmtutrnooh[VLLHL=BHN[^][ZVNB;<======rtw����������vtqhnqr{���������������{{{{���
##&%"
���������������������������������������������������������������������������<CFGGE@<4/#  #/<����


����������@BKNR[\\[Z[[[NJGB@@@��������������������#0<IMURQLI<0#[[hhjlkha[[Z[[[[[[[[
#/<?<<<4/,#
SZafnz��������haRILS�������nt��������vtnnnnnnnn&)+-5;BKNTYZNMB5)(&&MNS[\gt���ttgb[NMMMM)/5<?BD</'))))))))))+/9<HUV[ZUQH</,'%'++�W�L�_�c�c�o�����������������������s�g�W������|������������������������������������������Ŀοѿݿ���ݿֿѿĿ��������/�&�#��#�/�6�<�H�U�Z�^�V�U�H�<�/�/�/�/�����������������������������������������y�w�m�w�y�������������������y�y�y�y�y�y�����������	���	��������������������������q�Y�J�@�:�G�T�m�����Ŀѿ����ݿĿ���������"�;�G�T�`�j�`�T�G�;�.�"��	����������� ��������������M�@�4�-�)�'�'�.�4�@�M�Z�f�m�o�m�i�f�Y�M�b�X�Y�]�^�a�b�n�z�{ŇŉŉŋŇ�{�u�n�b�bŭūŢťŭŹž����������������������Źŭ�E�B�E�e�p�x���������ûϻлǻ����x�_�S�E��������
�����
���������������������	����۾�����	��!�)�.�/�.�)�"��	�ѿʿĿ����������Ŀѿڿݿ���������ݿѹ��ܹϹƹĹϹܹ���������������Ŀ����������ĿѿҿݿݿݿֿѿĿĿĿĿĿĿ���ƾ��ľ׾���	��$�,�+�'�&� �%�"������������������������������������������y�y�y������������ĿοȿĿ����������y�yìçÛÓÇ�z�w�v�zÃÇÓ×àìòûùöì����������������)�2�5�L�O�F�5�)�����|�s�r�s�s�����������������������������������������}�k�Q�G�H�O�s��������������`�V�\�`�m�y�������������y�m�`�`�`�`�`�`ĿĽĽĿ�����������
���"����������Ŀ��������(�2�4�9�?�4�(�������<�9�9�<�H�O�U�V�^�\�U�H�<�<�<�<�<�<�<�<�������"�'�/�9�;�<�;�4�/�"�����L�I�H�L�Y�e�r�~���������������~�r�e�Y�L�����������ºɺ���!�-�9�:�-�!���ֺ��M�B�A�H�Z�s��������������׾������s�M�g�`�Z�W�T�Z�g�s�y���������s�g�g�g�g�g�g�����������������������¾Ⱦ�������������ƳƨƧƣƧƳ��������������������������Ƴ�������������$�)�'�$������������������ܻɻû����ûллܻ�������������ù������������ùŹϹܹ����������ܹϹ�����������������'�(��������ŹųŭŚřŠŭŹ����������������������Ź�Ŀ¿����Ŀƿѿܿݿ޿�޿ݿӿѿǿĿĿĿ�EEEED�EEEEE*E7ECEPEPENECE=E7E*E�������������������ɺԺ��!�C�6���ֺ������������
�
���
��������������������"����	�������������
��%�-�7�?�@�<�"�������"�#�*�+�"���������D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxD|�0�)�$��#�$�0�=�I�P�O�I�>�=�0�0�0�0�0�0�����������������	�
������
�������f�Z�_�f�o�r��������������������r�f�f��������������(�4�:�G�K�<�4�!�������������������������������������������������������������������������������������a�T�L�H�C�B�D�H�R�T�a�m�s�y�������z�m�a��������������
��)�6�B�L�M�B�:�)����à×ÚÕÚàìôõñìéààààààààÇ��z�n�b�a�`�a�b�n�zÇÑÓàááàÓÇ�U�R�H�=�<�8�9�;�<�<�H�I�Q�U�Z�_�U�U�U�UE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF!F$F'F+F)F$FFE�E�E� 4 J C 8 B 4 . Z n I @ ; 5 B ? 2 p U ) J j e _ ; s U L > 7 ) $  J j = ] D . Q 9 F c � : l z y [ U ' T g 2 j * U � c - z ) )  �  �  �  �  �  u  R  �    X  ,  L  �  |  :  �  0  �  p  �  F  T  q  %  r  �  �  �  �  q  �  �  S  !  �  �  z  �  F  :  �  m    U  �  e  �  �  �  �  �     �  T  �  F  .  �  V  �  J  F  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  �  �  �  �  �  |  ^  9    �  �  �  �  �  |  ^  1  �  �   w  �  �  �  �  �  �  p  L  %  �  �  �  �  `  D  '     �   �   �  X  b  f  b  X  J  7  !  	  �  �  �  �    T  )  �  �  �  l  �  �  �  �  �  �  �  �  �  �  z  ^  B  '    �  �  �  y  <  \  b  g  i  j  g  a  U  G  4      �  �  �  �  w  `  s  �  �  �  �  �  z  s  i  _  P  ?  (    �  �  �  �  �  z  T  +  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  y  u  p  l  g  T  L  =  ,    �  �  �  �  w  Y  F    �  �  D  �  �  8   �  [  N  C  K  Q  P  M  H  A  6  -  %    �  �  �  G     �   �        �  �  �  �  �  �  �  �  �  �  �  �  |  y  w  t  q  	�  
�  U  �    C  x  �  �  �  �  �  =  �    
,  	%  �  �  P  p  �  �  �  �  �  �  �  q  O  4      �  �  �  �  c  '    ?  P  Z  ]  S  @  (    �  �  �  �  P    �  �  �  �  �  y  �  �  �  �  �  l  G    �  �  q  3      �  �  �  �  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  �  �  �  �  �  �  �  �  �  �  s  X  :    �  �  C  �  &  �  �  �  �  �  s  _  K  5      �  �  �  �  �  h  6  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  t  h  c  j  p  �  �  �  �  ~  t  j  _  T  H  =  1  &     �   �   �   �   �   �  w  q  ]  @       �  �  �  	          �  �  �  Y  0  �  �  �  �  �  �  �  �  �  �  �  �  {  s  r    �  �  �  �  �  �  �  �  �  �  t  g  X  H  7  !    �  �  �  �  p  @     �  �  �  �  �  �  p  e  Z  I  7  &      �  �  �  �  �  e  <      
    �  �  �  �  �  �  �  �  �  �  �  r  /  �  Y  �  �  �  �  �  �  �  �  �  �  �  ~  l  Z  G  4        �   �   �  �  �  �  n  i  ^  J  ,    �  �  �  _  6    �  z  �  U    e  _  Y  S  J  A  7  +      �  �  �  �  �  �  �  �  �  �  X  U  T  R  h  _  H  '    �  �  �  L    �  �  \  �  2  F  �  �  �  �  �  �  �  �  �  �  p  [  E  .    �  �  �  �  �  �    B  a  x  �  �  �  �  �  �  �  t  c  R  ?  *    �  j  b  _  \  Y  P  F  <  0  "      �  �  �  �  ~  ^  :     �  $  /  2     	  �  �  �  �  `  1     �  �  G  �  �  I  �  =  �  �  �  �  �  �  �  Y  $  �  �  r    �  i    �  A    �  /  6  7  -    
  �  �  |  R  f  �  5  �  -  �  �    �  �  �  �  �  �  �  �  �  �  o  S  6    �  �  �  �  S     �   �  �  z  a  H  /    �  �  �  �  �  �  p  Z  D  -  !  G  n  �  O  A  3  #    �  �  �  �  �  {  Y  5    �  �  �  O  �  k  �  �  �  �  �    p  a  L  6    �  �  �  �  ^  2    �  y  �  �  �  �  �  �  �  �  �  �  �  �  w  _  B  #  �  �  �  �  �  �  �  �  �  �  �  �  �  _  .    �  �  �  �  d  *  �    U  F  7  (             �  �  �  �  �  �  F    �  K   �  `  a  d  ~  �  ~  l  X  @  $     �  �  x  ;    �  u    �  a  a  a  ]  P  C  3  !      �  �  �  �  �  z  V    �  8  Z  3  N  H  9  4    �  �  l    �     b  �  
�  	�  �  �  �  �  �  a  3    �  �  �  �  z  R  )  �  �  �  F    �  �  �  �      -  =  N  _  `  Z  S  L  E  ?  8  0  (  !      
  �  �  �  �  b  R  1  -  %    	  �  �  �  \    �  O   �   �  |  r  g  ]  S  I  >  3  )          �  �  �  �  �  �  z  
I  
0  
  	�  	�  	�  	h  	!  �  �  �  *  �  ;  �  �  "  :  F  �  W  T  J  >    )  F    �  �  �  K  
  �  �  7  �  r    �  T  ;    �  �  �  s  E  -    �  �  u  A    �  �  k  �  �  �    +      �  �  �  �  �  Z  -  �  �  �  f  +  �  p  D  X  P  B  *    �  �  �  �  f  B  #    �  �  u    �  N   �    	  �  �  �  �  �  �  �  z  [  <  	  �  �  H  
  �  �  T  U  @  )    �  �  �  �  ]  8    �  �  �  y  H  �  �      �  �  
    �  �  �    T  $  �  �  X    �  G  �  f  �  /  U  ,    �  �  u  0  �  �  s  .  �  �  _    �  E  �    a   �  �  �  �  }  Q  &  �  �  �  �  r  M    �  �  G  �  �  "  �  !    �  �  �  �  �  �  h  G  )    �  �  �  �    M    �  �  �  �  �  }  n  _  O  @  /      �  �  �  �  ~  L    �  m  a  V  J  <  -    
  �  �  �  �  �  }  g  P  =  <  :  9  �  �  �  �  �  �  �  e  1  �  �  \    �  �    �  �  �  �