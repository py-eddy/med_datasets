CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����l�D      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�1�   max       P�4[      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =�Q�      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(��   max       @E�p��
>     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vt�\)     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q`           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @���          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >L��      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�0�   max       B,}R      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��H   max       B,�      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?t   max       C���      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C��      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�1�   max       PM�       �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���E��   max       ?�n��O�<      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       =���      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(��   max       @E�z�G�     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vt�\)     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @M�           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @��           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?'   max         ?'      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?rn��O�<   max       ?�n��O�<     �  K�                     k   T                  &   !   :                        6         
         	         F      
      t            	               
   '      �      NQ��N^VINI�AM�1�OY�,Nf�P�4[P��NjeqNNQN_�:N���O���P��P��Pv��NmpkPh"RO�WOI�#O�P<�N��Pi�<N�o�N4�PN��fN9�pO�`�N���O1?�N��jPf�N��N��%Nُ:P3��N�+�Ny�8N~�OYɠNM81N���NV��O�*Os$O�h,O�~O�N�N�� NqM-�ě��u�t��ě���o��o;D��;D��;��
;��
<o<o<D��<D��<T��<T��<T��<T��<T��<T��<e`B<u<�C�<�C�<�C�<�t�<�t�<�t�<���<��
<�9X<���<���<�`B<�`B<�`B<�h=o=�w=�w=#�
=#�
=#�
='�=P�`=P�`=T��=�%=�%=�+=�Q�59<<FIRTPJI<55555555��������������������	)))(								@>?BOQQOEB@@@@@@@@@@��������

������cegstv������tgcccccc�����)/BNY\YH���zz��������������zggimt����utsgggggggmqt�������ytmmmmmmmm!)1-)#+/<HSTQH></#���
#/<ALLH</#�������)6ZVQB6�������������"& ��������5Bm|���QC5)�3445BCIJNWNB?5333333rv�����������������r��������������������suy���������������ws������������������������
#<UaUHT<#
�����������������������@Wen������������tSD@V\achkmnz�����zynnaV��������������������MOTUabmoz}ztma]TMMMM'')15>BBB<5)'''''''''/<HTagmkfTH/+%(&����������������������������������������-467;BO[`[YPOB=6----x������������������x--//;<=HRUZWURH<2/--dfgfnz��zxndddddddd������������������������)6AKO[dOB����!)6BOPROHB6)$��������z�������������������������������������	�����������������������������������������������������������������������������������������  ���	!26BLIEC>6)[XVX]afmqz{|~{zoma[[����������

���������


	���������.--/<DHTRJH<7/......��������������޽����������)�6�=�<�7�6�3�)������������6�B�E�J�C�B�<�6�)�(�)�*�6�6�6�6�6�6�6�6�l�y���������y�t�l�e�l�l�l�l�l�l�l�l�l�l�����������������������������������������	�������������������������������H�f�w�����z�m�_�H�/�"�������������[�g�t�{�[�N�)�������������<�[ÓØàìõììàÓÎÇÅÇÓÓÓÓÓÓÓ������������������ú�������������������ҾZ�f�h�h�f�a�Z�M�G�I�M�S�Z�Z�Z�Z�Z�Z�Z�ZD�D�EE
EEEEEED�D�D�D�D�D�D�D�D�D�`�m�y�������������y�`�T�G�@�?�?�D�T�_�`�	�"�;�H�X�Q�H�;��	�������������������	�ݿ�������%�*�(����ݿĿ����ĿѿݿT�`�m�y�������y�T�O�;��	�ݾѾؾ��"�T���(�/�+�(�����������������4�M�d����������������f�Q�A�(�����4�;�T�]�`�j�k�h�`�^�T�G�B�;�3�.�%�%�.�2�;�Z�f�s����������������s�f�Z�A�;�5�A�M�Z������������������������������������/�T�Z�\�T�H�5�$��	�����������������6�>�C�H�O�`�\�O�G�C�A�6�*�����*�0�6�Z���������������������s�A�9�(�"�'�5�A�Z�������(�5�6�;�9�5�(������������ϹϹٹܹ������������ܹϹϹϹϹϹϹϿѿݿ����޿ݿѿѿпĿ¿��Ŀ̿ѿѿѿ��[�g�k�t�{�t�k�g�f�[�U�N�[�[�[�[�[�[�[�[�T�a�m�r�z���s�j�a�T�H�;�)���"�/�;�H�T�������¼������������������������������������нݽ��нĽ������������z�|���������лܻ��������������ܻۻһл̻лллл����@�Y�l�g�Y�4���ܻû����˻ͻڻ������������� ��������������������E�E�E�E�F
FE�E�E�E�E�E�E�E�E�E�E�E�E�Eٺ~�����������������������~�v�s�x�~�~�~�~������������ּɼ��������������������ʾ˾׾޾�޾ؾ׾ʾƾ����������ʾʾʾʾ��
�	������������
�����
�
�
�
�
�
�
�/�<�H�S�S�H�@�<�/�(�&�-�/�/�/�/�/�/�/�/���û˻лػ׻ջлû������������������������������������������������������������������������������������y�v�o�o�m�s�y����������%��������������������������a�n�zÇÓàãàÙÓÇÁ�z�n�a�Z�I�U�U�a��#�0�<�F�I�R�U�L�I�<�0�)�#�������e�r�~�����������ú����������r�_�P�Q�Y�eŔŠŭŹ��������������ŹŭŠŞŔœŐŔŔDoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DuDiDeDo�n�z�~�z�t�s�o�n�a�V�U�I�U�X�a�l�n�n�n�n���������������������������������������� a D 1 P  J 2 S + = U K 1 a O I k - = ^ J ` i O k [ C E " [ v d S " H + : @ � 8 / O p G Y = I >  f _    |  �  c    �  �  �  �  v  l  �  )  \  �  �  R  �  �  $  �  Y  �  �    6  b  �  K  i  �  �  �  $     �  �  %  
    ~  �  s  �  r  j  H  R  7  �  �  ���t��#�
��`B�D��<49X;o=�;d=� �<#�
<#�
<T��<�9X=�P=H�9=8Q�=�\)<u=0 �<��<��=��=t�<���=�hs<�j<�1<�/<�j=C�<�h=\)=�P=\=P�`=�P=�w>t�=#�
=<j=T��=H�9=49X=T��=T��=�hs=y�#=�^5=��>L��=���=ě�B&�B5 B�BgRB��B	��BAB��B	�9B
*vB->B�UBB5#Br�B��B��BXxB��BA�B�B!�B١B
yB�B!a�A���B.�A�0�B"�B Q@BUVB!0B��BڈB��Bs
B��BI�B�>B��B)t�B,}RB;�B�8BaPB0aA�%�B�$B)�B��B&C�B	�B9�BJ�B��B	�KB@�B8�B	��B
?'BC�BO�B,B�LBBA^BU<BE�B>B>�B>�BͬBB
��B<�B!��A���B?�A��HB"@B M�B@5B!A�B�tB��B}�BF�B��BH`B��B�B)C2B,�BAVBCBF)B?�A���B�BB?-B?5A/txA֨�A��SAJA�mA���A�� A��<A�4EAϙ]A>��C�O�Aj�A���A�`�AgcA3��A@n�Ad�LAA��A�s�A�'qA��kA��A�Z?tA{�rA�CyA���@��A#��@�x�@�*�A�fC���@.@�IAQ\�A�K�A�wb@�>A rA�?`H�A�VA맨@��A��fC�ЭAƨnA���A-%�A��A�؇A�A�#A��A���A�j�A��AςA>�C�I�Ah�A��A�Af�A3nBAD�Ae�AAK�A�~�A�wA��BA���A��U?�A{�A��A��n@��A%Q�@��@���A�}�C��@�	@��AP��A�~�A�i
@��A"�A �?P .A��A�@��A�yC���AƀA怕                     k   T                  &   !   :                     	   7         
         
         F      
      t   	         
                  (      �                           ;   ;                  7   '   ;      1            -      7                           /            +                                                                                    )   !   #      -            '      -                                                                                 NQ��N^VINI�AM�1�N�FNf�OtI�O��NjeqNNQN_�:N���O}�PH�O��XO���NmpkPM� N��+Ow�N8JO���N~'P(�N�o�N4�PN���N9�pO�`�N���O1?�N��jO�N�&~N��%N��Oҕ�N�+�N4.�NP`GOYɠNM81N`TkNV��Ow}Os$O*
�O�~O%��N2�NqM-  �  �  Q  K  :  (  	4  �  E  t  _  �  C  �  2  7  �  �    �  �    �  �  �  �  {  �  e  �  �    k  7  �  �     �  �  �        S  t  �  �  �  �  �  F�ě��u�t��ě�;o��o=�7L=D��;��
;��
<o<o<�C�<ě�<���=o<T��<u<�1<�o<���<���<�t�<�/<�C�<�t�<���<�t�<���<��
<�9X<���=]/=o<�`B<�h=�%=o=#�
='�=#�
=#�
=8Q�='�=T��=P�`=�o=�%=���=�O�=�Q�59<<FIRTPJI<55555555��������������������	)))(								@>?BOQQOEB@@@@@@@@@@������� ����������cegstv������tgcccccc � )158:;80) ��������������������ggimt����utsgggggggmqt�������ytmmmmmmmm!)1-)#+/<HSTQH></#��
#/<>IH@</#
������)<A>6)��������������
�����	BNgktvnb[LB5)3445BCIJNWNB?5333333yz�����������������y��������������������}wx{���������������}��������������������������
#<PVG</#�����������������������KHN^n������������tSKV\achkmnz�����zynnaV��������������������OPTalmwqma^TOOOOOOOO'')15>BBB<5)'''''''''/<HTagmkfTH/+%(&����������������������������������������-467;BO[`[YPOB=6----��������������������//3<HMUVUSLH<6//////dfgfnz��zxndddddddd�������������������������)6BHIGB)���!)6BOPROHB6)$���������������������������������������������	�����������������������������������������������������������������������������������������  ���)6>=>=:61)[XVX]afmqz{|~{zoma[[��������


����������

�����������.--/<DHTRJH<7/......��������������޽����������)�6�=�<�7�6�3�)������������6�B�E�J�C�B�<�6�)�(�)�*�6�6�6�6�6�6�6�6�l�y���������y�t�l�e�l�l�l�l�l�l�l�l�l�l����������	���������������������������������	��������������������������	��"�/�;�I�O�N�H�A�;�/�"��	� ������	���5�B�N�W�Z�M�4�)���������������ÓØàìõììàÓÎÇÅÇÓÓÓÓÓÓÓ������������������ú�������������������ҾZ�f�h�h�f�a�Z�M�G�I�M�S�Z�Z�Z�Z�Z�Z�Z�ZD�D�EE
EEEEEED�D�D�D�D�D�D�D�D�D�m�y�����������}�y�`�R�G�E�B�D�I�Q�T�`�m�"�/�;�J�L�H�=�/��	�����������������"�ݿ��������������ݿͿĿ��ĿݿG�T�`�l�����|�y�`�G�;�.�)�"���#�.�;�G���(�/�+�(�����������������4�M�f��������������������f�V�A�*��&�4�;�G�T�V�Z�\�W�T�H�G�;�5�.�-�.�2�;�;�;�;�M�Z�f�s��������������s�f�Z�H�A�<�A�H�M������������������������������������������/�;�G�S�U�R�9�+���	���������������6�9�C�E�C�>�6�*�����*�4�6�6�6�6�6�6�N�Z�g�����������������g�N�A�2�+�*�.�A�N�������(�5�6�;�9�5�(������������ϹϹٹܹ������������ܹϹϹϹϹϹϹϿѿݿ��ݿܿѿĿÿ¿ĿͿѿѿѿѿѿѿѿ��[�g�k�t�{�t�k�g�f�[�U�N�[�[�[�[�[�[�[�[�T�a�m�r�z���s�j�a�T�H�;�)���"�/�;�H�T�������¼������������������������������������нݽ��нĽ������������z�|���������лܻ��������������ܻۻһл̻лллл����'�4�A�I�I�@�9�4�'���������������������������������������������E�E�E�E�F
FE�E�E�E�E�E�E�E�E�E�E�E�E�Eٺ~�������������������~�y�u�z�~�~�~�~�~�~�������ּ����ܼϼ��������������������ʾ˾׾޾�޾ؾ׾ʾƾ����������ʾʾʾʾ����
����
�����������������������������/�<�H�O�P�H�?�<�/�*�(�/�/�/�/�/�/�/�/�/���û˻лػ׻ջлû��������������������������������������������������������������y�����������������y�x�u�s�x�y�y�y�y�y�y������%��������������������������a�n�zÇÓàáàØÓÇÀ�z�n�a�\�K�U�V�a��#�0�<�F�I�R�U�L�I�<�0�)�#���������������������������~�r�i�e�\�[�e�r�{��ŔŠŭŹ��������������ŹŭŠŞŔœŐŔŔD�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzDxD{D~D��a�n�q�q�n�h�a�\�U�N�U�\�a�a�a�a�a�a�a�a���������������������������������������� a D 1 P ! J " U + = U K 3 a V , k 0 / R B ^ 5 D k [ B E " [ v d <   H ) : @ d : / O f G V = 9 >  ; _    |  �  c      �  �  ;  v  l  �  )  �    �  8  �  h  �  ~  X  `  �  !  6  b  �  K  i  �  �  �    �  �  �  �  
  o  _  �  s  �  r  Y  H  e  7  ]  F  �  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  ?'  �  �  �  �  {  t  m  d  [  Q  H  ?  6  /  +  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Q  K  E  @  :  4  .  )  #          �  �  �  �  �  �  �  K  E  @  :  5  /  *  #          �  �  �  �  U  #   �   �        &  ,  3  6  9  :  9  6  -      �  �  �  T    �  (  #          	    �  �  �  �  �  �  �  �  �  �  �  �  �  _  �  w  �  )  x  �  �  	   	-  	3  	2  	  �    4  +  l  �  d  �  F  �  �  #  W  �  �  �  �  �  �  �  f  �  1  W  l  �  E  :  .  #      �  �  �  �  �  �  �  �  u  e  M  4      t  j  a  X  N  A  4  '       �  �  �  �  q  P  (      �   �  _  ]  \  Z  V  J  >  2  !    �  �  �  �    a  @     �   �  �  u  _  G  -    �  �  �    O    �  �  Z    �  �    "    0  <  B  B  9  )    �  �  �  M    �  �  '  �  4  �   �  G  n  �  �  �  �  �  �  �  �    g  E    �  �  f  #  �  ?      *  1  0  +            
  	  �  �  �  F  
  �  �  �  �  �  �    )  7  5  &    �  �    G  �  �  �  C  �    �  �  �  �  �  �  �  �  �  �       $  6  H  Z  l  ~  �  �  o  �  �  �  ~  x  j  P  )  �  �  �  �  �  �  �  d    �  F  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  Z  y  =  �  �  �  �  �  �  �  �  �  �  �  �  z  e  @    �  �  "  �  �    .  <  [  �  �  �  �  �  �  �  �  �  �  �  p    �  (  �  �              �  �  �  �  �  �  �  �  M    �  j   �  u  �  �  �  �  �  �  �  �  �  �  �  x  Q  )     �  �  �  Z  �  �  �  �  �  �  �  �  �  �  h  :    �  �  N  �  m  �  H  �  �  �  �  y  k  \  N  ?  1      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  H  `  y  u  n  g  `  M  5    �  �  v  2  �  �  �  B     �  �  �  �  �  �  �  �  �  �  �  �  �  w  o  g  _  V  M  D  ;  e  P  @  5  ,         �  �  �  �  T  #  �  �  �  f  3   �  �  �  �  �  �  �  �  }  v  l  _  R  D  4     
  �  �  �  �  �  �  �  e  H  -    �  �  �  �  �  �  �  �  y  ^  3    �      �  �  �  �  �  �  �  �  �  _  7  �  �  t  6  �  �  &  V  �  �    �    U  g  i  U    �  C  �    �  x  �  �  �      '  3  5  /  $       �  �  �  �  v  R  2    �  �  �  �  �  �  r  a  r  �  �  �  �  �  �  �  �  �  �  �    l  Z  �  �  �  �  �  �  {  q  b  S  E  4  #    �  �  �  �  �  �  
�    {  �          �  �  `    
�  
  	�  �    �  �  S  �  �  �  �  y  i  V  C  -    �  �  �  �  �  W  ,     �   �  �  �  �  �  �  �  �  �  n  ^  N  ?  1  #           �  �  �  �  �  �  �  �  �  �  �  �  �  ^  ,  �  �  q  (  �  �  ,    	  �  �  �  �  �  o  S  6       �  �  �  �  �  {  `  D        
      �  �  �  �  �  �  �  �  �  �  ~  l  [  J  �  �  �  �  �  �  �      �  �  �  �    Q    �     �   -  S  W  Y  Y  K  2    �  �  �  q  9  �  �  y  3  �  �  Y    n  t  l  V  >  #  �  �  �  �  \  7    �  �  �  O    �  �  �  �  �  �  {  k  [  H  2      �  �  �  �  �  �  s  L  %  ^  g  Q  �  �  �  �  �  r  F    �  �  �  >  �    M  n  i  �  �  �  �  x  \  @  %    �  �  �  �  �  g  F  #  �  �  �  �  s    �  �  8  i  �  �  �  K  �  R  �  �  �  n  �  	�  R  h  �  �  �  �  �  �  �  �  �  Y  "  �  �  s  4  �  �  k  $  F  7  (      �  �  �  r  8      �  �  �  �  �  �  �  �