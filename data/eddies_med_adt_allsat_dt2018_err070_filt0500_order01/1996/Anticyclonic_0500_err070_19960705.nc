CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�n��O�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��z   max       P�r�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       >bN      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E��G�{     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @va�����     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @P`           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�`          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �D��   max       >bM�      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��,   max       B,��      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,�`      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��]   max       C���      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��!   max       C���      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��z   max       PL��      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e��ڹ�   max       ?����E�      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       >bN      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E�\(�     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @va�����     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P`           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B[   max         B[      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?��,<���     �  N�      +         	      	   *   #                        I   !            	            5   !      W      '      #            ;   %         �                        
      �         NK[O�PCN��N��pN��{N�.LN!��P$j�O��OO�*�O���O��O�#�Ncd�N�)�N�(�P�r�O�c	N�ȖOngEO�1N��P	�N�5�O���PE�OpkmO�6�O���N"oMP36O$\�O� NXWN8�N��P�P��N�8OF�2P_�tN�*�N�-N��M��zOg�O�Z�N>r�N�ѴNF��O�0�N*aN\NnN:�˻�`B���
�o�o��o;D��;�o;�o;��
;��
;��
;�`B;�`B<t�<u<�o<�C�<�t�<��
<�9X<�j<�j<�j<�j<�j<ě�<ě�<ě�<�/<�`B<�`B<�h<�h<��=+=C�=��=�w=�w='�='�=0 �=49X=8Q�=T��=T��=]/=e`B=ix�=�%=� �=�v�=�G�>bN������������������!+/<HJUafbXUA<//$!�����������������������������������������������������������525BNY[dgihg[RNBA555��������������������������)3<?FA7)��x���������������|x 06BJM[gj[OB8)bght�������������thb#030+-360#
 !#0<U\ffbUOI>0)# �������������������� "/;=;:/)"qnntv����������tqqqq����5B\_\\WNB5����}������������������}�� $)69;6)������������������0/5BNX[\[WQRNIIB@850FKN[gktx����tig[PNFF\bt��������������ug\���������������
#-/28@D<0&�����)HX`l]B5)���)5BEN[cgg[NB50){�����������������nmp���������������{n)0/)	#/<Han����n[H<#	fefhnt����������tphftu���������������}vt������� �����������$(/<HPHE</$$$$$$$$$$woz������������zwwww����)6>?;6)����������� ���������������������������,4;CHIOT[[ZVTKH:10/,�����):BKOB%������������������������gglt��������tlhggggg89?BBCIOOONB88888888'),5BBB5)���������������������������
,*������ #&),(%#
)/+)'% �����
��������������

����������������������������������������������������������������������������������������� �"�������������������������/�<�>�<�<�6�/�)�*�(�/�/�/�/�/�/�/�/�/�/����������������������������������������'�3�@�F�@�@�;�6�3�'����!�'�'�'�'�'�'������������������������������������������������������¿²®²³¿�������������˾��ʾ׿	�.�G�V�L�:�"��	�׾ʾ�����������ĦĿ��ĺĴĿķĦč�r�[�O�G�C�[�k�vāčĦ���������ѿֿڿѿĿ��������������z�{����������������ݽĽ������������ƽ������������������мɼʼּ���r����������������������o�f�`�Y�S�Y�d�r�����������������������������������������T�a�e�m�o�v�w�m�d�a�X�T�P�Q�T�T�T�T�T�T�@�L�Y�d�e�q�p�m�e�a�Y�L�H�@�;�<�@�@�@�@��<�U�l�p�a�b�n�b�<�0����������������������������������ñêð�����������zÇÓàíìëãàÜ×ÓÇÂÁ��z�z�t�z�(�5�A�N�Z�a�g�j�_�<�5�(�������!�(�����
�������������ùðù������������ÓàèêãàØÓÈÇ�z�z�u�z�|ÀÇÉÓÓ�4�A�V�f�z�y�n�Z�M�A�(��������+�4�:�F�S�_�l�m�x�|�x�l�i�_�S�F�:�.�.�8�:�:�M�Z�s������g�M�4������������A�M�ѿ�����'�.�-����ݿĿ����������Ŀ��h�tāĚİĶĳĭĞčā�s�s�u�r�j�h�c�^�h�Ŀѿֿݿ����������ݿѿ����������������O�[�h�n�n�j�h�^�E�@�6�)��
�������6�O�Z�b�b�_�Z�M�H�J�M�V�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�	�"�;�Q�Y�[�W�D�7�"�	�����������������	�������������������������y�s�p�q�s�{��"�.�J�O�P�G�B�;�.������ܾӾ׾���	�"�#�/�<�@�=�<�4�/�#�!��#�#�#�#�#�#�#�#�#��� �$�"�����	��������������������������������������$�/�9�>�?�:�.��	��������������������3�L�V�\�q�u�x�w�r�\�L�3���������l�x�������������������x�u�l�_�T�_�e�l�l��0�8�<�G�<�0�&��
�������������������4�M�^�m�u�{�{�s�Y�@�4�'�"�������4�N�Z�g�s�s�������s�g�Z�N�A�>�A�D�N�N�N�N���"�&�(�"�"��	����������	����������ɺֺٺֺɺ������������������������������������������������������������������n�{ŇŔśŠţŠŜŔŇņ�{�n�m�j�g�m�n�n�`�o�y�����������������{�l�`�S�G�E�G�O�`���	��"�.�9�.�"��	��������������������ƧƳ������������������������ƳƩƧơƧƧ����������������������������������������DoD�D�D�D�D�D�D�D�D�D�D�D�D{DoD`DVD^DfDo���%�$�����������������ǈǔǡǭǩǡǔǈǄǂǈǈǈǈǈǈǈǈǈǈEuE�E�E�E�E�E�EwEuEmEuEuEuEuEuEuEuEuEuEu 1 " z G ^ X z @ ; E J E $ 7 [ < - 0 \ F a \ 3 I � - G ) T a K ( 2 J c V R U Y J / e " c | 2 l p d d " < ( <  d  �  d  �  �  
  x    N  C  I  [  S  s  �  �    �  A    q  '  n    H  :    -  w  g    _  �  d  �  �  �  �  �  �  �  �  �  U  x  3    �    {  �  1  k  M�D��=��;��
;ě�;�`B<e`B<D��=<j=#�
<��
<���<u=\)<�o<��
=+=�E�=L��=o=�P=��<��=8Q�=t�=<j=���=e`B=,1=�l�=o=�+=P�`=�%=��=�w=8Q�=ě�=���=<j=ix�>A�7=]/=T��=L��=]/=��w=���=}�=�+=�7L>bM�=���=�>�PB��B�BB >�B#)B!=BY8B�oBsCB�#BЭB�4B%XB&O`B!�A��,B��B��B9�B��BQ0B��B	"�BjwB��B%�B��B�0B�B� BE�Bu�B�BȅB�{B��B�sB@�Bp�B+]!A�2�BH�B3�B
zB<�B�vB��B,��B��B,1BD�B
�B �B��B�FB�B��B �hB#=HB!>IB��B�-B�rB��B�"B��B%�B&<�B"5gA���B��B|�Bf,B��B6aB*�B	7�B�!B�B$�wB��B�B�bB��BUaB��B�\B��B��B�B;B��B�B+C�A��+B6<B?B
:`B@nBͼB�|B,�`B�,BCuBϚB>�B ;�B �BA�)�Aү�A�_�@�$�?��]A��7A�?lAXgA�ZeAs�PA+h�A��@�\�@�]A���?�Q�A�� AЭ�A�<A�Aх�A��A9�v@��A:2A~�A�-�Az(�A���A>�#A�dGAHDRA\]MA�w�A���A�q�A�sm?��f@�k�A詜@���A�MA�@-|aA�/�A�!A�A\�B�A��C��@�N)BV�C���A�fA҃�A��@���?�3wA��EA���AU aA�p�At�9A,(�A�@�"?@$�A���?טtA�!8AЙBAʀlA��Aњ�A�xA:��@��A:�mA�o�AމIAz��A׏%A?�A�$AGeNA\��A�|A���A�gA��`?��!@���A�\�@ь>A�%A��H@*�3A���A���A9A\�B+A�oeC��@�#�BA�C���      ,      	   	      	   +   $         	               I   !            	            5   !      W      '      #            ;   &         �      	                  
      �                                 /   %                        9   #               %      )   -         %      +      #            )   +         /                  %                                                                        )                  %      )   %               +                  !   %         #                                       NK[N��N��NS�M��cN�@N!��O��O�� O\�3O���Nմ$O��8Ncd�N�)�Ni@�PL��O�V�N�M=OngEN��N��P ��N��*O���P��O<��Om�O�6MN"oMP36N��O�{N9�yN8�N��O�.�O�mcN�8O��O���N�*�N�-N��M��zN���Od��N>r�N�ѴNF��OB�N*aN\NnN:��  �  �  �  �  �  e  �  �  �  �  S      t    �  �  4  $  �  �  �  K  �  �  �  H  A  �  �  L  �  �  �  �  �  (  �  �    H  �  e  �  �  �  *  �  �  �  �  �  �  ���`B<T���o��o;D��;�o;�o<���<D��;�`B;��
<t�<49X<t�<u<�1<�/<�j<�j<�9X<���<�j<ě�<���<�j=\)<�`B<���=@�<�`B<�`B=+=�P=o=+=C�=8Q�=@�=�w=49X=Ƨ�=0 �=49X=8Q�=T��=]/=m�h=e`B=ix�=�%>$�=�v�=�G�>bN������������������-+//./<GHQUXURH><:/-������������������������������������������������������������9ABNW[cghge[TNBB9999�������������������������#14860)���������������������#,6BEJ[]^[OB1)bght�������������thb	
#&)/030.#
		##'+0IU_cYSI<0-&#�������������������� "/;=;:/)"qpt{�������tqqqqqqqq������5BNWVTK5)�����������������������	)6686)						�����������������5235BMNV[[[YTNBB:555FKN[gktx����tig[PNFFadt��������������xla����� ��������
#-/28@D<0&���)5BNV]]XNB5��!&)+25BNQ[[`b[NB5)��������������������}z~����������������})0/)	#/<Han����n[H<#	ghiqt}���������tjhgg{yy{���������������{������ �������������$(/<HPHE</$$$$$$$$$$woz������������zwwww�����)6;<83)��������������������������������������53215;@HJQTXXXTTH;55�����$)*%�������������������������gglt��������tlhggggg89?BBCIOOONB88888888'),5BBB5)�������������������������������	����� #&),(%#
)/+)'% �����
���������������

���������������������������������������������������������������������������������������������������������������������/�<�>�<�<�6�/�)�*�(�/�/�/�/�/�/�/�/�/�/�����������������������������������������'�3�6�9�5�3�'�&�!�&�'�'�'�'�'�'�'�'�'�'������������������������������������������������������¿²®²³¿�������������˾׾����"�2�9�/�"�	����׾ʾ¾������ž�čĚĦĲĲĮĹĳĦĚčā�t�[�T�R�[�uāč�����������ĿҿԿѿĿ�������������������������������ݽĽ������������ƽ�����������������ڼּͼμּ����r�����������������������r�e�\�W�Y�f�r�����������������������������������������T�a�e�m�o�v�w�m�d�a�X�T�P�Q�T�T�T�T�T�T�L�Y�_�e�k�j�e�Y�N�L�B�D�L�L�L�L�L�L�L�L���
�#�<�I�d�g�Z�Y�\�I�0�#�
��������������������������������ùñ÷ÿ��������ÇÓàæáàÙÔÓÇÄÄÂÁÇÇÇÇÇÇ�(�5�A�N�Z�a�g�j�_�<�5�(�������!�(�����������������������������������ÓàèêãàØÓÈÇ�z�z�u�z�|ÀÇÉÓÓ�4�A�U�f�x�w�l�Z�M�A�(��������(�4�F�S�_�h�l�x�y�x�l�`�_�S�O�F�:�7�:�@�F�F�M�Z�s������g�M�4������������A�M�ݿ����$�&�!�����ݿѿſ��������Ŀ��tāčĖĚĦĬĲĩĦĚĚčā�x�w�t�m�h�t���Ŀѿݿ��������ݿĿ����������������B�O�[�d�a�`�U�B�<�6�)�"������)�6�B�Z�b�b�_�Z�M�H�J�M�V�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�	�"�;�Q�Y�[�W�D�7�"�	�����������������	��������������������������}�t�v��������	��"�.�8�E�;�5�)��	�����������	�/�<�?�<�<�2�/�#�"��#�'�/�/�/�/�/�/�/�/��� �$�"�����	�����������������������������������������/�4�:�;�5�&��	���������������˺�3�@�P�U�a�j�k�e�Y�L�3�����������l�x�������������������x�u�l�_�T�_�e�l�l�����
��#�.�0�4�0�#���
��������������4�@�M�X�d�h�h�c�Y�M�4�'��������4�N�Z�g�s�s�������s�g�Z�N�A�>�A�D�N�N�N�N���"�&�(�"�"��	����������	����������ɺֺٺֺɺ������������������������������������������������������������������n�{ŇŔŚŠšŠŚŔŇŁ�{�q�n�k�h�n�n�n�S�`�l�y�~���������������y�l�`�S�M�I�I�S���	��"�.�9�.�"��	��������������������ƧƳ������������������������ƳƩƧơƧƧ����������������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D{DoDfDmDoDzD{���%�$�����������������ǈǔǡǭǩǡǔǈǄǂǈǈǈǈǈǈǈǈǈǈEuE�E�E�E�E�E�EwEuEmEuEuEuEuEuEuEuEuEuEu 1 # z , O J z A 6 E J 7 ' 7 [ : $ # h F ? \ 1 L �  < ) M a K # ) I c V 6 F Y F * e " c | " N p d d ' < ( <  d  �  d  ]  3  �  x  �  �  �  I  �  �  s  �  {  h  3  �    �  '  E  �  H  L  �  �  d  g        K  �  �  �  �  �  P  �  �  �  U  x  
  	  �    {  �  1  k  M  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  B[  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  �  )  `  �  �  �  �  �  �  �  �  t  :  �  �  >  �  G  �  �  �  �  �  u  S  /  
  �  �  �  �  �  �  r  ^  I  4      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  r  f  s  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  d  e  a  T  C  0    �  �  �  �  y  W  6    �  �  �  �  Z  �  �  �           �  �  �  t  I    �  �  �  c  2  �  �  �  �    >  Y  p  }  �  ~  s  c  P  6    �  u    �  0  �  �  �  �  �  �  �  �  �  �  �  S    �  s  F  �  {  �  t  <  �  �  �  �  �  �  �  �  �  �  �  {  a  @    �  �  �  _  1  S  O  H  ?  3  #    �  �  �  �  �  r  I    �  �  �  W  !                    �  �  �  �  �  �  �  x  g  ^  U  �  �        �  �  �  �  �  U  '      �  �  {  K  g  T  t  k  a  W  K  >  1  "      �  �  �  �  �  �  �  �    ^               �  �  �  �  �  ~  ^  =    �  �  �  i  8  {  �  �  �  �  �  �  �  �  �  �  v  N    �  �  F  �  �  >  �  �  �  �  �  �  �  p  U  F  =  *    �  �  %  �  M  �  k  �  (  2  4  ,  "      �  �  �  �  �  t  =  �  �  X  �  r  �  �  �  �  �  #  !        �  �  �  �  8  �  �  .  �  T  �  �  �  �  �  �  ~  a  A    �  �  �  �  �  h  T  R  e  �      X  z  e  N  3    �  �  �  T    �  �  i  -  �  �  �  �  �  �  �    g  L  0    �  �  �  c  U  H  :  ,      �  J  K  I  B  8  +      �  �  �  �  �  �  b  1  �  �  K  !  �  �  �  �  �  �  �  �  v  \  B  '  
  �  �  �  �  �  �  l  �  �  �  �  t  T  A  C  M  9  #  	  �  �  �  �  u  j  f  �  �  �  �  �  �  �  �  �  �  �  z  q  Y  0  �  �    w  �  |  �  $  C  9    �  �  �  �  Q    �  �  @  �  e  �  �  '  �  4  >  A  @  :  1  '    	  �  �  �  �  {  U  1      �  �  
�    X  �  �  �  r  k  ^  G  .    
�  
y  	�  	<  K  +  �  q  �  w  n  f  ]  T  L  C  :  2  )           �  �  �  �  �  L  J  D  :  +    �  �  �  �  R     �  �  �  R    �  R  
  �  �  �  �  �  �  �  �  �  h  J  &  �  �  [  �  �  6  �  �  �  �  �  �  �  �  �  �  �  �  �  y  M    �  �  8  �  h  �  w  {      w  p  e  W  J  ?  6  +      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  e  H  5    �  �  l  2  �  �  �  R  #  �  [  �  '  $      �  �  �  K    �    +  �  >  �  �    i  �  �  �  �  �  �  �  �  g  -  �  �  g  -  �  �  "  �  �   �  �  �  �  �  �  �  �  y  \  :    �  �  �  �  T  (   �   �   �  �  �  �        �  �  �  �  �  �  p  E    �  �  t  2  �  
�  �      ~  �     E  5  �  �  @  �  4  `  
u  	f      �  �  �  �  �  �    s  e  O  7      �  �  �  T    �  �  ?  e  Y  M  @  4  (        �  �  �  �  �  �  �  �  �  w  a  �  �    s  f  W  G  8  (      �  �  �  �  �  �    m  \  �  �  �  �  �  �  �  �  �  �  �  �  {  t  m  f  _  X  P  I  a  �  �  �  m  P  /  
  �  �  |  ;  �  �  4  �  #  �     k  �  �    *         �  �  �  �  ^  +  �  �  =  �  i  �   �  �  �  �  {  n  `  R  B  0      �  �  �  �  �  w  X  9    �  �  �  �  �  h  N  2    �  �  �  �  �  u  c  Q  <    �  �  �  �  �  �  �  �  �  �    r  d  W  I  ;    �  �  B     w  O  �    V  |  �  �  _  .  �  _  �  �  �  B  �    B  	<  �  �  }  r  f  [  P  E  ;  3  ,  (  1  ;  F  S  a  p  �  �  �  �  �  �    a  B  $    �  �  �  �  `  <    �  �  j  '  �  �  �  �  x  X  7    �  �  �  ~  ;  �  �  w  8  �  �  k