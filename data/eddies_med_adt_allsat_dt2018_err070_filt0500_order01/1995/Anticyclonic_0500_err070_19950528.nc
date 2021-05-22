CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�&�x���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nغ   max       Pփ�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��o   max       =��      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E��Q�     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
=    max       @vw��Q�     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�y�          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >.{      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��G   max       B,i&      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��r   max       B,C8      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�!*   max       C�r�      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��M   max       C�X      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nغ   max       Po�w      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��E����   max       ?�w�kP��      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       =���      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E�
=p��     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vw��Q�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�@           �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?ye+��a   max       ?�u��!�/     p  S(                     #                  %   �   E   	   	   V   !                           2      h         R         	      >         @   
   �         	   ,                  *      "            1N��O�N�USO���OƑjN�P�P�lNH�6O	�P<��O6q�N.<wPi�P�[Pe$)No��N8�Pփ�O��]O��'O�:O��SO�!N;|�N��OnN��PfOg^�P�ƒNn�HOH�=Pc�kO1*uO��N��
N�nP��O��:NغO�O�O�+
N�]�N��N�7wO�
N�&�O%��O�D�N��
OQdOڤ�O0A�Oe�pO"L�NУSOe��O����o�u�u�e`B�#�
�o��o��o��o��o��o��o;�`B;�`B<t�<#�
<e`B<e`B<e`B<u<�o<�t�<���<���<�1<�1<�9X<�j<�j<���<���<�/<�/<�/<�<�<��<��=\)=\)=�P=�P=�P=��=��=��=49X=49X=49X=8Q�=8Q�=H�9=Y�=]/=e`B=u=�7L=��w=����������������������ffgltw����������tjgf��������������������9:HUanz�������znUH>9TUXan�����������naXT� #"������etw��������������tie���������������������������������������������������������������������������������������������������+5VQVSN=:1)<;>FN[gt��������tNB<�����+5?;)�������$')6ABNOOONDB64)$$$$��������������������KVa�����//�����aK��������������������.5BN[g``[PB5)vv���������������cgt��������������sgc"10;HNHNNLHB;/"QT[]hntih[QQQQQQQQQQ^^ajmoz����zma^^^^^^����������������������������������������
#2;IUbkjcI0!	
�
�����
#%/155/#
��;Dh����������voOAG7;���������������������������������������������)5FQRVVNE5��MVdht���yvtjh_^[TPM��������������������##	�}{|����������������������
#'++#
�����5BN[tv~�~ytmg[NE=;5596:<HKRNH<9999999999()-04;GTV`fkmgaTH;/(���������������������������

�����>?HJT[akefa]TH>>>>>>������

���������Y[\`hmty|�uthe][YYYY���������������������������������������� #/8<AHLKHD?</+& �����)4:973,)��	$%#
)59@CCB?5)
}zz���������������z}����������������������������	����������������������������������������������������

�����������)-00+���������������������������������������zÇÉÓÞàãàÝÓÏÇ�z�x�n�i�g�n�t�z�=�I�V�`�b�o�t�o�b�b�V�I�=�0�0�-�0�7�=�=���������������������������������������E�E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E�E��������������������u�s�p�s�{�����������������������������������������������������������)�5�9�5�)������������������Y�Z�T�O�L�B�3�'��������'�3�@�L�Y�N�g�������������������s�Z�N�.�%�!�!�2�N��������������������������������������������������������������������������"�G�T�q���y�w�m�`�T�;�5��	�������"�)�B�[�tĂĈĆ�}�l�T�D�6������� �)�;�H�M�a�z���z�m�T�H�3�&������������"�;ÇÓÕßÓÑÓÔÓÓÇ�z�y�x�zÂÇÇÇÇ�O�T�W�S�O�B�6�4�6�6�B�E�O�O�O�O�O�O�O�O�����������"�=�F�I�5�	�����������o���������(�5�A�N�P�_�g�i�f�Z�N�(�����������%�5�=�H�H�K�A�5�(�����޿ܿ߿���/�<�H�L�P�L�H�B�<�8�/�$�#���!�#�-�/�/������������������������s�n�l�l�s������"�/�;�I�T�a�o�w�m�T�;�/�"��	�������	�"�H�U�`�a�i�a�U�H�G�A�H�H�H�H�H�H�H�H�H�H�Ŀѿ׿ݿ߿ݿտѿƿĿ��������ĿĿĿĿĿĿ�"�.�3�;�A�@�B�;�.�"����	�����ìù��������ùíìàÓÇÄÇÇÓàèìì�������ʼ���������f�V�Y������������`�m�y�������������}�y�m�`�G�<�?�G�H�T�`�������������Y�4���ܻû����л޻���Y������!�*�+�!�����������������*�6�<�C�K�O�R�O�N�C�*���������*�\�hƚƧ�����������Ƴƚƀ�u�k�b�Q�S�\�:�F�_�d�j�_�S�F�B�-�!�������!�-�:���������пѿݿ�ݿѿοĿ����������������"�.�;�?�?�;�0�.�"����������	���"��"�/�;�H�I�N�H�E�;�3�/�"� ���������	��"�*�.�1�1�(�"��	������������������i�g�[�N�B�5�.�)�&�$�*�5�N�[�t����������������������������������������������#�0�@�I�B�<�#���������������������(�5�>�A�N�N�D�A�(���������
��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzD�D�D��n�{ŅŇŏŔŕŔŇ�{�r�n�d�b�n�n�n�n�n�n�/�<�B�@�>�<�/�)�#������#�%�/�/�/�/�����������������������}������������������)�=�I�B�5�.�������ÿù���������h�tāčĒėĕčā�t�l�h�[�U�[�e�h�h�h�h�f�s�������������s�f�Z�T�M�A�6�A�M�Z�f�#�0�<�I�U�\�a�b�U�I�0�#��
���
��� �#����������������������������������������������������������������¿°°»¿�����ؼ4�@�M�f�r����w�f�M�@�4�'��������'�4���#�$�%�%�#�������������������
����!�-�:�H�S�V�S�J�:�!������׺�����y�����������������y�s�l�j�f�]�_�`�l�q�y���ʼϼּۼ����ּʼǼ���������������E�E�E�E�E�E�E�E�E�E�EuEiEgEgEaE`EjEuE�E��L�e�����������������~�r�e�L�'����,�L 4 ) 5 C & # R � R / 1 M @ ` [ M F W > R Y E d W ; 6 q S  q ?  O H W l / , > " D S @ O > 2 8 4 L $ $ 0 C F _ \ Q F c    �  M    �  �  �  "  �  o  .  �  R  �  �  )  �  n  �  N  �  B  �  }  h  �  9  �  �  �      �  2  �  �  %    �  p  3  �  |  �  �  �  �    �  �  d  �  �    �  9  �  �  �  7�t�<D����o;�`B<��㻣�
<��%   <u<�h<T��:�o=49X>C�=��-<���<�9X=���=@�=#�
=t�=��=L��<���<���<���=8Q�=��=49X>�=��=0 �=�;d=8Q�=0 �=��=0 �=�j=u='�=���=@�>.{='�=P�`=<j=�-=�%=ix�=�+=ix�=��-=\=�O�=�E�=���=���=��>+Bf�B
3�BX�B��B�gB�,B��B�B �B�B�$BJ�B�=B	+'Bg4B��B�B�TBrBAB��B.A��GBn�A��6B!v�B"P�B&�B;�B�HB 5�B��B�B�1B_�BVWB
�aB�*B��B6A���B�/B. A�^�B��B�BˍB=BX:B ZB�QB��B�BL�B+�B,i&B��B�4B�
B�pB
@B?�B�B?�B�BBB@]B ��B��Bx'B2\B�<B	>B��B��B�B��BC�B��BB
�|A��rB�A��oB!u�B"@B%�2B1�B� B ?�B��B� B��B��BIEB
�B�$B�aB��A��gBAB?�A�W#B��B��BڲB?�B��B��B�:B�TB¿Bg�B�B,C8B�LBʗB�LA��A�96Bl$A��dC�r�A��8A�w�A��?�!*A���B��A��#Ab�VA�;A��AɮFA�ܥA��DA�M�A��"A��AH�A�ŊA�P�Ay�(A_a�A�q@�Aj�@�:�@c�A��)B1_@�Av�!A^��A�{A��}A� �A�e�A�$A�.�C���A�tA��AI
UA�AA�	AA<�A���A�C}A���@ϫA�m@hiA��@���C��K?鮟A���A�zmB>�A��C�XA�'A��PA�1�?��MA��SB��A��Ad�A�}sA�6vA�~*A��A�McA��kA��1A�AG�jA�{�A��Ay�UA_[=A��@�W�Aj�@�S�@dA�j}B?�@f�Aw�A_ �A�f�A�hA�[�A�wA�r�A�NC��A�hA��AH�;AҼ�A�K�AAJ�A�i�A�l�A��Z@��A�~�@o]�A�@��C��`?�o�                     $                  &   �   E   
   	   W   "                            3      h         S         	      >         @      �         	   ,                  +      "            2               !      +         -         '   )   5         K      !         !               1      C         3               %         !                  '                  &                  %               !      !         )         %      '         =                                                                                       '                  #                  %N��N�R[N�USOj�\OƑjN�P�O�T�NH�6NjeSP�XO#�N.<wO�vOY��O���NB�N8�Po�wO��]O�ޙN�#�O��SO(�N;|�N��OnN��`Opm�Og^�O~��Nn�HO'�O���O�mO��NbfMN�nOQp�O�<NغO8�ZO�O�J�N�]�N�V�N�7wO�
NězO%��O�D�N��
O9ɲO�G�N�y�Oe�pNå-NУSOY�	O�P  i  �  :  M  �  �  �    �  '  �  �    �  �  �  �  �  B  �  
  �  Y  �  O  �  W    �  
�  �  9    �    p  �  L  *  �  
+  �      "  �  ^     �  H  |  S    �  q  ]    Y  ���o�49X�u�t��#�
�o;��
��o;D��;��
%   ��o<T��=ix�=C�<49X<e`B='�<e`B<���<�1<�t�<�/<���<�1<�1<�j=49X<�j=���<���<�h=�7L<�`B<�=o<��=}�=�P=\)=y�#=�P=}�=��=�w=��=49X=8Q�=49X=8Q�=8Q�=P�`=e`B=ix�=e`B=�o=�7L=���=�����������������������nhiot���������tnnnn��������������������ECMU^anz{�����znaUHETUXan�����������naXT� #"������yuz����������������y���������������������������������������������������������������������������������������������������5CKOKB75)�JFGINN[gt|�����tg[NJ������#$������%()6@BLCB65)%%%%%%%%��������������������zu������#�����z��������������������)BNZ[XSMDB5)��������������������cgt��������������sgc "/6;>HHIHC=;/"QT[]hntih[QQQQQQQQQQ^^ajmoz����zma^^^^^^����������������������������������������!#0@IQUY[WOIB<0#�����
#%/155/#
��NKKP[htxy}|uthe[SPN����������������������������������������

)5=BFGGE>5)MOV[eht~����}xth[VOM��������������������! �}{|������������������������

�����?;8=BR[dgs|}xtg[NG?96:<HKRNH<999999999985469;BHT[_bcaaTHF;8����������������������������

���>?HJT[akefa]TH>>>>>>������

���������Y[\`hmty|�uthe][YYYY���������������������������������������� #/8<AHLKHD?</+& �����)4:973,)��	$%#
)58?BBB=5)��}}��������������������������������������������	������������������������������������������������������

������������)./)��������������������������������������n�zÇÓÚàààÚÓÇ�z�r�n�k�j�n�n�n�n�=�I�V�`�b�o�t�o�b�b�V�I�=�0�0�-�0�7�=�=����������������������������������������E�E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E�E��������������������u�s�p�s�{�����������������������������������������������������������)�5�9�5�)������������������3�>�@�D�@�:�3�'�����'�(�3�3�3�3�3�3�g�����������������s�g�Z�N�5�(�'�5�<�N�g��������������������������������������������������������������������������G�T�d�m�w�r�m�`�T�;�(��	�����"�.�G�6�B�O�[�h�i�s�r�h�[�S�O�B�6�1�-�+�+�0�6��"�;�T�_�n�p�n�h�a�T�H�;�-�	��������ÇÓÔÞÓÏÇ�z�z�z�zÃÇÇÇÇÇÇÇÇ�O�T�W�S�O�B�6�4�6�6�B�E�O�O�O�O�O�O�O�O�����������!�3�<�0������������������������(�5�A�N�P�_�g�i�f�Z�N�(��������������+�?�@�5�(������������/�<�H�K�H�H�>�<�3�/�*�#� �!�#�%�/�/�/�/������������������������s�n�l�l�s������	��"�/�;�?�E�H�K�H�=�;�/�"��	�����	�H�U�`�a�i�a�U�H�G�A�H�H�H�H�H�H�H�H�H�H�Ŀѿ׿ݿ߿ݿտѿƿĿ��������ĿĿĿĿĿĿ�"�.�3�;�A�@�B�;�.�"����	�����ìù��������ùììàÓÇÅÇÈÓàêìì����������ɼ���������������x�r�n�g�r��`�m�y�������������}�y�m�`�G�<�?�G�H�T�`�M�Y�f�r�~�~�w�f�Y�@�4�'������'�4�M����!�*�+�!�����������������/�6�C�E�K�M�F�C�6�*���������)�/ƚƧƳ����������������ƳƧƚƎƆƃƄƎƚ�:�;�F�O�_�c�i�_�U�S�F�D�:�-� ���!�5�:���������пѿݿ�ݿѿοĿ����������������"�.�4�:�.�+�"���
��!�"�"�"�"�"�"�"�"��"�/�;�H�I�N�H�E�;�3�/�"� �����������	����"�%�"���	�����������������N�g�t�|�t�r�g�[�N�B�5�0�'�&�,�5�N�����������������������������������������������
���#�)�%��
��������������������(�5�>�A�N�N�D�A�(���������
��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��n�{ŅŇŏŔŕŔŇ�{�r�n�d�b�n�n�n�n�n�n�/�<�@�?�<�<�/�&�#������#�.�/�/�/�/�����������������������}������������������)�=�I�B�5�.�������ÿù���������h�tāčđĖēčā�t�m�h�\�g�h�h�h�h�h�h�f�s�������������s�f�Z�T�M�A�6�A�M�Z�f�#�0�<�I�U�\�a�b�U�I�0�#��
���
��� �#����������������������������������������������������������������¿³³½¿�����ؼ'�4�@�M�Y�f�o�~�t�f�M�@�4�'��� ���'���
�� �"���
�������������������������!�-�:�H�S�V�S�J�:�!������׺�����y�������������y�o�l�k�c�l�n�y�y�y�y�y�y���ʼϼּۼ����ּʼǼ���������������E�E�E�E�E�E�E�E�E�E�E�EuEhEhEbEbElEuE�E��L�e�r��������������~�r�e�Y�L�3�'��4�L 4 # 5 > & # T � 4 # 0 M B % _ H F ] > M S E 0 W ; 6 o "  V ?  ' C W P / ! : " 9 S = O ? 2 8 * L $ $ - E , _ ^ Q F _    �      �  �  �  �  �  y  q  l  R    �  �  i  n  )  N  :  �  �  r  h  �  9  �  �  �  (    c    \  �  e    �  3  3  �  |    �  �  �    �  �  d  �    �    9    �  �  �  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  i  d  _  Y  P  H  =  2  &    �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  b  ,  �  �  R  �  }    �  �  {    :  6  /  !       �  �  �  �  �  k  Q  5    �  �  �  k    A  D  D  G  L  M  J  D  =  5  .  (      �  �  �  :  �    �  V  +  �     y  u  ^  =    �  �  �  m  2  �  �  �  �  `  �  �  �  �  �  �  �  �  �  �  w  n  e  \  S  J  B  9  0  (  n  �  �  �  �  �  �  l  R  ,      �  �  �  R  �  �  �   �        �  �  �  �  �  �  �  �  �  �  �  �  �  y  f  S  A  �  0  >  J  Y  p  �  �  �  �  t  d  T  D  &    �  w  4  #  �      $  &      �  �  �  Y  $  �  �  X     �  �  �  +  �  �  �  �  �  �  �  �  �  p  T  3    �  �  �  d  +  �  �  �  {  t  n  g  a  [  T  N  H  @  8  /  '           �   �  �  �  �    �  �  �  �  �  {  z  h  J    �  �  G  �  2  <  +  �  �    �  |  �  �  �  �  �  O  �  b  �  �  V  	�  �  �  �    L    �  �  �  �  �  �  �  �  �  G  �  �    S  L  �  �  �  �  �  �  �  �  s  P  *    �  �  t  W  >  1  $      �  �  �  �  �  �  �  �                     �  �  �  �  L  �  �  �  �  �  �  }  C    �  :  �  x  �    �  �  �  B  7  *      �  �  �  �  �  m  4  �  �  ]  �  �    �  <  U  �  �  �  �  �  �  r  [  =    �  �  �  7  �  �  s    �  �  �  �  �    	  �  �  �  �  w  F    �  �  e  "  �  l  !  �  �  �  �  �  �  ~  p  _  L  7       �  �  �  M  �  �  f  F  �  
  C  S  Y  O  4    �  �  �  9  �  ~    �  �    &  �  �  �  �  �  �  �  �  �  �  �  �  s  c  L  5    �  �  �  O  F  >  5  -  $        �  �  �  �  �  �  �  �  }  l  Z  �  �  �  �  �  �  �  �  �  �  �  �  ~  l  Z  H  4    	   �  O  U  I  1    �  �  �  k  _  #  �  s    o  �  4  �   �   >      *  1  3  +        z  k  J    �  �  ,  �    8    �  �  �  �  �  {  e  K  ,    �  �  W    �  �  �  n  W  y  �  0  ~  �  �  	�  	�  
(  
k  
�  
�  
�  
F  	�  	     +  �  S  >  �  �  �  �  �  u  e  V  G  1    �  �  �  e  !  �  �  C  �  /  5  8  8  4  /  '       �  �  �  x  P  %  �  �  �  H  �  �  �  (  d  �  �  �  �      
  �  �  f  �  W  �  �  !    x  �  �  �  �  �  �  w  e  P  6    �  �  �  a  *  �  �  �          �  �  �  �  ~  ]  ;    �  �  �  �  �  �  �  `  9  N  c  l  n  p  o  n  j  e  [  M  >  *    �  �  �  `  *  �  �  �  �  �  �  �  �  �  v  V  2  
  �  �  �  ^  0       �  �  -  >  F  I  H  H  G  J  L  C  "  �  �    a  �  s  *  "  *  '    �  �  �  �  �  f  A  '    �  �  f    �  L  R  �  �  �  �  �  �  �  �  �  �  �  r  Y  E  <  2  +  1  7  =  V  	  	y  	�  	�  
  
!  
*  
(  
  	�  	�  	n  	  �  �  :  f  �  �  �  �  �  �  u  f  X  N  E  >  *  
  �  �  �  t  F     �   �  k  \  �        �  �  W    �  M  �      �  /  	P    �        �  �  �  �  �  �  �  �  �  �  �  �  �  �    %  ;                   �  �  �  �  }  ^  ?  !    �    �  �  �  �  �  �  �  u  i  [  M  >  -    
  �  �  �  m  :    ^  A    �  �  m  %  �  �  <  �  �  B    �  �  ;  �  m  i  �     �  �  �  �  �  �  y  P    �  �  ~  G    �  �  R  �  �  �  �  �  �  �  �  �  r  ^  G  .    �  �  �  �  r  a  d  H  A  3       �  �  �  �  �  j  T  :      �  �  ~  B    |  {  y  u  o  f  U  F  7  '      �  �  �  x  1  �  F  �  H  P  R  J  =  -      �  �  �  r  D    �  �  =  �  �    �    �  �  �  �  m  1  �  �  X    �  T     �  `    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  Y  >    �  �  <  q  d  Z  Z  f  d  F    �  �  y  B    �    8  �  �  S  �    +  =  Q  \  ]  Z  V  O  D  5       �  �  �  s  T    �      �  �  �  �  �  �  }  e  F  &    �  �  �  v  ?    �  U  X  S  =    �  �  �  �  �  d  ;  
  �  y  $  �  R  �  �  e    t  Z  :    �  �  �  d    �  �  ,  �  j  �  E  `  �