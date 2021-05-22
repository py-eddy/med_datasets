CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�������      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�N�   max       P�ܽ      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       >	7L      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @F~�Q�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @v�p��
>     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @���          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       >333      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��k   max       B3�      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A� �   max       B3�4      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?@V   max       C��"      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C��      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          d      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�N�   max       PR�S      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�j   max       ?�0U2a|      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       >	7L      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F~�Q�     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @v�p��
>     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @��          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?i   max         ?i      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?�.��2�X     �  M�               	                                 
      #            7   $   G   "   "         %                     	         -      (   
                     d            M�N�N�$�NeSsN;�fN��N}e�N}(O��Nƪ�O��N�tAP9��N�|�N3YNV&O.ZO.bP%L�N�~tO�]Of��O��aP��P�ܽP$?mO��O�bN/wZP;N��O�E'O�N��$O�wON��lO(��ODI�N8��O�u"OlxO���O� N޿zNP�OPHRN��N���N'h�OqS�O?DqN���N�P�N#�9�t��+������1��1�49X�t���`B��`B%   ;�o<t�<t�<#�
<49X<49X<49X<T��<T��<u<�o<�o<�C�<�C�<�C�<�C�<��
<�j<���<�/<�<�<��=o=t�=��=��=8Q�=<j=<j=<j=@�=D��=m�h=y�#=y�#=�O�=�hs=���=�-=���>$�>	7Lmlnz{|�ztnmmmmmmmmmmTSQSU]bgnpwyunmbZUTT����������������������������������������SSUalnyyunha[USSSSSS//1<FHSRH<4/////////vwz���������vvvvvvvv'!%)/BN[gc_][NB52'��������������������RPP[ahkt��������th[R#!$)35<ABGDB5)######��������<HM/
�������������

���������������������������yz��������~zyyyyyyyy��������������������+()+/5<HJU[aVUHG</++-6GUan�������z_UH<3-mnuz����������|zvunm���
#%*.,/0/%#
��+'&&(/;?HTVWVVTPH;0+������  	
��������O[d��������������vdO������,251+���������)6BORRVMB)���������

����TQMUXanz~��znaUTTTT�������������������������5KKB95)"����x~������������������ 
-HUkqtnaU<) �������������������������������������������������������������:7<>HLUVXUSH><::::::����)121))����deaadgpt��������tpgd����������������������������������������3,+-45BN[egmrsg[NB53v������������������v��������	
���������������������������[[[dhtw���toh[[[[[[[99;@FQTacchkjdaTHE;9;<<INUWZUUI<;;;;;;;;&)4697:962,)���������������������������	��������[`_[VOD6)%-4@BOS[NUUabnwz~~zncaYUNNNN�������� ������������������������������Ͼ��������׾ξ׾����������㻅�������������������������|�x�o�x�������0�=�I�P�K�I�C�=�;�0�/�,�0�0�0�0�0�0�0�0āāăčĘčā�t�h�g�h�o�tĀāāāāāāE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E7E7ECEMECE>E7E*E E E*E7E7E7E7E7E7E7E7E7�ܹ����������ܹعѹܹܹܹܹܹܹܹܿĿѿݿ�����������ݿѿȿ��������Ļлڻһлɻû����������������������ûϻоM�Z�f�o�s�w�s�g�g�f�\�Z�W�M�D�A�>�>�A�M�׾������������׾Ҿ˾;׾׾׾׾׾��"�/�T�z���������v�s�j�a�N�/�"�����"����������������������������������������¿����������¿²­²´º¿¿¿¿¿¿¿¿�s�u���������z�s�g�c�g�q�s�s�s�s�s�s�s�s�����������������������������n�s�x���������������������������������������A�Z�g�~�����b�a�Z�A�(��� ���'�"�(�A�����������������������������������������m�u�y����|�y�m�`�T�L�G�C�;�:�G�M�T�c�m�	��"�/�;�F�J�H�E�;�/��	�������������	��������̼μʼ���������r�f�M�>�8�R�f��g�m�t�~�}�t�g�[�N�B�)�������B�g��/�;�H�k�q���n�H�/���������������������ʿ	�T�`�g�T�D�/�$�	����׾�������������(�4�;�G�[�f�p�s�f�Z�M�4�����������#�/�2�<�A�?�=�<�/�#������������(�.�*�(���������������0�I�U�f�k�n�m�h�b�U�L�<�0��
�������0���ʾ׾ؾ׾־ʾ������������������������������������~�t�f�Z�A�-�!�&�7�M�f�s�����ù������������ùìàÓÑÊÓÕßàìõù�������½������������������������������������$�1�=�<�8�0�$�����������������F=FJFTFVFYFVFMFJF=F4F1F+F1F5F=F=F=F=F=F=ƁƎƖƚƜƥƥƠƚƎƁ�u�t�q�p�p�u�xƁƁ�T�`�m�y�������������������y�l�_�U�T�N�T��!�+�-�:�D�:�-�!������������U�a�zÇÓàæëßÓ�z�a�H�4�0�8�J�P�R�U�������������������������������������������������������g�N�K�K�E�C�E�J�Z�g���������
��#�&�%�%�#���
�������������������������'�%����������������������'�4�4�@�K�E�@�5�4�3�)�'�$�&�'�'�'�'�'�'�0�<�E�U�b�n�u�z�n�b�U�I�<�0�*���� �0���������������޺��������⻑�����������ûƻû����������������������������"����������������D�D�D�D�D�D�D�D�D�D�D�D�D}D|D�D�D�D�D�D��ɺ����������������������������ֺݺٺֺɺY�e�e�q�r�z�r�o�e�Y�Q�N�Y�]�Y�W�Y�Y�Y�YEuE�E�E�E�E�E�E�E�E�E�EzEuEqEuEuEuEuEuEu���������������������������������������� v 3 7 X L = A : n B % I d e ? 6 - ; [ B @ 6 2 * _ I ! B 4 E � D m 9 * 9 G v 4 > ; / + P R R V � O n a ; |    "  �  �  w  �  �  �  [    =  �  v    c  8  �  :  �    g  �    r  e  y  f  .  B  |    �  9  �  �  s  {  �  |  �  �    "     �  �  M  "  �  �  �  �  �  ��C��������㼋C��D��:�o�D��<�t�%   <o<#�
=�P<e`B<�o<u<�1=t�=D��<�t�<�h<��=�hs=T��=�-=P�`=P�`=t�<���=�%<�h=Y�=u=t�=y�#=]/=@�=m�h=L��=�^5=���=�{=ix�=q��=�hs=��P=�C�=��T=���>333=��=�S�>�u>�PB4�B'��B~sB��Bo�B�B��B�B"��B�)B.GB�[B�`B��B M�B�MB��B��B��BC�A��kB"��B
�UB=�BmB#��B�"BK�B�B3�BőB!�mB��B�qBAB�yB
>�B*k�B��B,"B��Bq�B;�B�A�7�B&�BRB�fB�B��B�MB}FBp�BD�B'�6B��BI�BARB� B��B>�B#A�B;�B1B��B�RB��B @|B{�B��BP`B��B@A� �B"�B
�B�MB=OB#�3B�BC]B@bB3�4B��B">�B�EBB6B@?BA�B	�B*��B�B$�B��B��B^�B�aA��}B&��B=�B��B�.BDB�BB��BF�AU#e@�ÔB
ȸA�G�C�2�C��?@VA��@��XA>�SAU�A��A�S�A�lA���A��:A�U�A�^fA��Ah�RA�:@�$A��A�V%AX�bA8��A���A�x�A�R�AP�!AC0�Ǎ�A!l�A�	�C��"B�*Am�F@o�ZA��%A���A�S{A���A2EG@͢�A�Wm@J�@���?P&%C���@"e'?�$�C���@�AAS%�@�B
��A�6~C�8C��4?��A�}#@� �A= �AT��A�A�}^A�~dA�a$A���A��A��KA�o�Ai�A��@��A��A�۟AU��A6��A�|�A��wA�AP�A@�|A�~�A$�)A�{�C��B��Am�@t&�AȁfA��uA�kA���A1�@��A뗛@K�{@��1?U�VC��@'ߒ?�y�C��@�               
               	                        $            8   $   G   #   #         &                     
         .      (                        d                                                1                  -            '   %   9   /            %      -         !               !      !                                                                        #                  /               !   1   '                  -         !                                                         M�N�N���NeSsN;�fN��N}e�N}(N��1Nƪ�O��N�tAO�ȓN�|�N3YNV&N��N�$4P�GN�~tO�]O>'�O��O��PR�SO��Oq�.N��N/wZO�vzN��O�E'NҏN��$O�wON��lO(��O��N8��O��+O&�OV�O� N޿zNP�O;�&N��N���N'h�O�=O?DqN^s�N�P�N#�9  �  �  R  �  �  &  �  ,    ?  z  �    R  �  	  �  �  �  6  �  �  �  S  C  �  @  /    q  �    �  �  '    "  O    1  �      p  '  �  �    �    �  7  _�t���������1��1�49X�t�;�o��`B%   ;�o<�o<t�<#�
<49X<T��<��
<u<T��<u<�t�<�`B<��
<��<�j<�9X<�1<�j=t�<�/<�=C�<��=o=t�=��=#�
=8Q�=]/=T��=q��=@�=D��=m�h=}�=y�#=�O�=�hs=���=�-=���>$�>	7Lmlnz{|�ztnmmmmmmmmmmTUWbjnqnnlfb_XVUTTTT����������������������������������������SSUalnyyunha[USSSSSS//1<FHSRH<4/////////vwz���������vvvvvvvv4356:BNW[]\\[TNEB@54��������������������RPP[ahkt��������th[R#!$)35<ABGDB5)######�������-!
��������������

���������������������������yz��������~zyyyyyyyy��������������������///3<HLULH<//////////:HU_n�������znbUH</mnuz����������|zvunm���
#%*.,/0/%#
��-*()-/;HQTUUUTTH;4/-�����������������g_[g��������������qg�����$..+$����

$+<IMRSOB6
�������

�����USSUYanz|��~znaUUUUU���������������������������#12-%����x~������������������ 
-HUkqtnaU<) �������������������������������������������������������������:7<>HLUVXUSH><::::::����)121))����ccegrt���������~tjgc����������������������������������������0/35BNX[`ggjge[NB750����������������������������	
���������������������������[[[dhtw���toh[[[[[[[9:;AGSTabbfjicaTH<;9;<<INUWZUUI<;;;;;;;;&)4697:962,)���������������������������

������[`_[VOD6)%-4@BOS[Xacnvz}}zndaXXXXXXXX�������� ������������������������������Ͼ��������׾ξ׾����������㻞�������������������x�u�x���������������0�=�I�P�K�I�C�=�;�0�/�,�0�0�0�0�0�0�0�0āāăčĘčā�t�h�g�h�o�tĀāāāāāāE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E7E7ECEMECE>E7E*E E E*E7E7E7E7E7E7E7E7E7�ܹ����������ܹعѹܹܹܹܹܹܹܹܿݿ���������������ݿӿѿпѿڿݻлڻһлɻû����������������������ûϻоM�Z�f�o�s�w�s�g�g�f�\�Z�W�M�D�A�>�>�A�M�׾������������׾Ҿ˾;׾׾׾׾׾��/�;�H�T�a�����z�l�a�H�;�/�"�����"�/����������������������������������������¿����������¿²­²´º¿¿¿¿¿¿¿¿�s�u���������z�s�g�c�g�q�s�s�s�s�s�s�s�s���������������������������}�������������������������������������������������A�Z�e�w���y�_�]�N�A�(������$�+�'�A�����������������������������������������m�u�y����|�y�m�`�T�L�G�C�;�:�G�M�T�c�m�	��"�/�;�>�E�C�;�2�/�"��	����������	�r������������ü���������r�d�M�N�Y�b�r�B�N�g�r�|�}�x�p�g�[�N�B�)��	��	��)�B��/�H�X�d�f�e�W�H�"�	������������������ʾ�	��.�G�M�;�.�"�	���׾ƾ��������ʾ�(�4�B�T�Z�f�h�f�Z�M�(��������
����#�/�0�<�A�?�<�<�/�#������������(�.�*�(���������������#�0�<�I�U�Y�_�b�e�b�U�I�0�#��
����#���ʾ׾ؾ׾־ʾ������������������������������������~�t�f�Z�A�-�!�&�7�M�f�s�����ìù����������ùìâàÚÓÍÓØàãìì�������½������������������������������������$�1�=�<�8�0�$�����������������F=FJFTFVFYFVFMFJF=F4F1F+F1F5F=F=F=F=F=F=ƁƎƖƚƜƥƥƠƚƎƁ�u�t�q�p�p�u�xƁƁ�m�y�������������������y�o�m�b�`�Y�`�c�m��!�+�-�:�D�:�-�!������������a�zÇÓßãàÙÓ�z�a�U�H�A�E�H�I�R�Y�a�����������������������������������������s�����������������s�g�Z�S�N�L�L�S�Z�g�s���
��#�&�%�%�#���
�������������������������'�%����������������������'�4�4�@�K�E�@�5�4�3�)�'�$�&�'�'�'�'�'�'�0�<�A�U�b�n�r�s�n�b�U�I�<�0�,���#�%�0���������������޺��������⻑�����������ûƻû����������������������������"����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ɺ����������������������������ֺݺٺֺɺe�n�r�y�r�m�e�Y�R�O�Y�_�e�e�e�e�e�e�e�eEuE�E�E�E�E�E�E�E�E�E�EzEuEqEuEuEuEuEuEu���������������������������������������� v N 7 X L = A 9 n B % % d e ? $ + > [ B A * * ) L F  B  E � D m 9 * 9 F v ' 8  / + P P R V � 1 n E ; |    "  �  �  w  �  �  �      =  �      c  8    �  �    g  �  R    v  Y      B  p    �  �  �  �  s  {  g  |  J  n  �  "     �  �  M  "  �  R  �  �  �  �  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  ?i  �  �  �  �  �  |  v  q  k  f  ^  T  K  A  7  .  $        �  �  �  �  �  �  �  �  �  �  �  t  ]  G  /     �   �   �   �  R  Z  a  f  d  c  ^  V  N  E  =  4  '    
  �  �  �    M  �  �  �  �  �  �  �  �  |  o  `  R  C  4  %     �   �   �   �  �  q  `  L  7  .  )  "    
  �  �  �  `  "  �  �  \     �  &          �  �  �  �  �  y  T  )  �  �  ~  8  �  �  Y  �  �  �  �  �  �  �  �  �  x  k  \  M  3    �  �  ]     �  �  �  �        &  *  ,  *    �  �  �  q  6  �  �  A  �        �  �  �  �  �  �  �  p  e  �  �  �  �  �  �  �  �  ?  4  )                 	  �  �  �  �  �  �  �  �  �  z  q  i  `  V  L  B  5  '      �  �  �  �  �  k  Q  7    �  �  �  �  �  �  �  �  �  �  ^  ,  �  �  �  T  D    �  m            �  �  �  �  �  �  �  �  �  �  �  �  ~  t  k  R  C  5  &      �  �  �  �  �  �  y  ]  ?       �  �  �  �  �  �  �  �  �  �  x  k  ^  M  8  $    �  �  �  �  ^  5  �  �    	    �  �  �  �  �  �  �  |  g  P  9    �  �  i  �  �    6  V  s  �  �  �  n  Y  >    �  �  w  4  �  �    s  �  z  c  J  ;  B  @  ;  6  $    �  �  m     �  d  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  z  6     	  �  �  �  �  �  �  �  �  �  �  y  d  L  ,  �  �  J  �  �  �  �  �  �  �  �  x  b  I  )    �  �  �  _  0    �  \  �  �  �  �  �  �  �  �  �  i  -  �  �  *  �    �  ]  |  ^  z  �  {  q  a  J  (  �  �  �  E    �  �  J  	  �  o  �  �    <  O  R  E  %     �  �  p  4  �  �  }  :  �  i  �   �  8  >  >  B  :  !  �  �  �  �  |  Y  (  �  �  {  R    �  t  �  �  �  �  �  �  x  p  ^  A    �  �  r  ,  �  �  +  �  L  =  ?  /    �  �  �  �  �  t  a  W  O  @  �  H    �  �  i  /  $        �  �  �  �  �  �  �  |  g  M  3        �   �  �  �  �  �        
  �  �  �  �  �  S    �  o    �  �  q  l  f  a  [  V  P  K  E  ?  9  1  )  !      	     �   �  �  v  \  ;    �  �  �  �  �  �  o  [  I  ,    �  m  �    �          �  �  �  �  i  9    �  �  `  (  �  �  l    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  s  O  (  �  �  �  ~  J    �  �  |    �  2  �  '      �  �  �  y  ?  �  �  h    �  V  �  �    �  3  �    �  �  �  �  �  �  }  b  I  2      �  �  �  �  �  +   �  �    !          �  �  �  �  �  �  {  \  5    �  7  �  O  >  -      �  �  �  �  �  �  �  �  m  R  6     �   �   �  �                   �  �  �  S    �  >  �    !    �      -  0  -  $      �  �  �  s  :  �  �  >  �  %      `  k  �  �  �  �  �  �  �  �  �  V    �  a  �  ]  �  �    �  �  �  �  �  �  �  �  �  {  p  e  Z  L  =  (    �  �      �  �  �  �  �  �  �  z  X  3    �  �  �  Q    �  p  p  n  a  J  .    �  �  �  J    �  �  \    �  �  K    �  "  %  &  #    �  �  �  �  �  w  Z  :    �  �  �  q  L  &  �  �  �  �  }  c  J  2      �  �  �  �  �  x  ^  @  "    �  �  �  �  �  ~  a  @    �  �  �  e  6    �  �  �  �  �    �  �  �  �    b  E  )    �  �  �  �  �  ~  l  \  L  ;  �  �  $  �  �  �  �  �  ^    �  �  N  �  �  �  �  �  �  �    �  �  �  �  �  y  U  0  Z  /    �  �  �  z  0  �  6  z  �  �  �  �  �  �  �  �  �  |  a  E    �  �  B  �  �  .  �  7    �  �  �  v  <  �  �  o     �  �  :  �  �  '  �  X  �  _  /  �  �  �  �  �  ]    �  �  I  �  �  a    �  l  *  �