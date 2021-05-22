CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�\(�\      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �49X   max       =      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?8Q��   max       @E�ffffg     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vYG�z�     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @���          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >�%      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,�F      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =���   max       C��      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =|�H   max       C���      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         m      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       PL      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?�@N���U      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �49X   max       ><j      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?8Q��   max       @E�          H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @vYG�z�     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @��          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @   max         @      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�R�<64   max       ?�?|�hs     �  M�         
         0   	        m   $      ?   6      /                           ?         �   	      :   P   D                  
      ]      2         +            !      	   N�k�Nm�NL��N/�=Om��P��Nz��N"�N��P���O�؏N*�6PO��#O�l�PkcvN���Os6�O��LN���N,׈O>.Nj��Oa� Pt�O�Or'|O�5NXSMO�_qP��6PP.nO�RORN�O�OB;N�\�N1&wO<�MO�4O2W|O���O˅2N$O�S�N�KAN�N�uO��N�+N��Na���49X�49X�ě���j:�o;o;o<t�<D��<T��<u<�o<�C�<�C�<�t�<���<���<��
<��
<��
<��
<�1<�j<ě�<���<�/<�/<��=o=o=+=C�=C�=t�=��=�w=8Q�=8Q�=<j=@�=D��=L��=L��=T��=]/=m�h=m�h=}�=�o=��-=��w=�-=����������������������������������������cdhnt����thcccccccc��������������������##,/<HUWaikcaUH</+##^Taz��������������t^��������������������#''#")5>BLNPONB5)""""""&$*;N[t��������t[B/&��������������������248;>HJLH;2222222222������� ��������������������������=;>HO[hktwzxtkhg[QB=�"5[gond]WSN5�LNNRY[ginqnkg][NLLLLY\kt������������tf[Y���� 
#/49<>>0#��~|��������������~~~~�����������������������)4520-)%�%%#&)5BNONB?5,))%%%%�����������dabht�����������{xod�����
#%*($#
���+)/<HUanxz�zsnaUH</+�����
!#"
��������������������������MYcnt����������t[TNM������&6<?<6)��������)BO[a`XK9*���?==BEOg�������t[OME?��������������������LJLN[^^[PNLLLLLLLLLL������������������yvwz{�������������~y��������������������))0*)������������������������������������������������������������������	��������������������������������


������������zz�����������������),-+)	|�����������||||||||����

���������� 	
#/652/*#
��������������������UV`amxzzzrmaUUUUUUUU��������������������ĦĳĿ������������ĿĳĦğĞĦĦĦĦĦĦ������
���
��������������������������L�Y�]�e�l�g�e�Y�N�L�F�K�L�L�L�L�L�L�L�L��������������������������������������������%�(�#��������������������������������������������g�(�����(�Z���#�/�<�>�C�<�9�/�#���!�#�#�#�#�#�#�#�#�ܻ����������ܻ׻ܻܻܻܻܻܻܻܻܻܿ`�m�y�q�m�g�m�m�m�`�^�[�U�Y�`�`�`�`�`�`���6�O�]�b�d�b�[�O�6�����������������4�@�O�R�M�A�B�4�����ֻܻѻ�����'�4���(�*�(� ���������������/�U�a�q�v�t�l�a�H�/�
�������������
��/���4�A�R�f�q�s�t�f�Z�(�����������������׾߾�Ͼ������f�M�?�4�*�4�K�f���ƎƤ������� �������ƳƚƁ�r�]�g�uƎ�)�5�7�B�N�U�T�N�B�5�)������)�)�)�)��"�/�3�7�=�<�;�5�/�"�� ��������	�����(�1�5�=�D�J�M�5��������������������ù˹Ϲ׹۹Ϲù���������������������'�3�7�6�3�'����������������������Ŀ˿пѿֿѿĿ������������������m�z�����������������������z�m�l�m�m�m�m�/�;�G�P�V�T�G�;�.�"��	��������	��"�/���ݽ���������ݽĽ����}�y�~���������`�m�y�������������y�m�`�\�T�L�L�T�\�`�`�������������������}�s�o�g�e�c�e�r�|����D�D�D�D�D�D�D�D�D�D�D�D{DoD_DSDVD]DoD�D��a�m�y�z�����z�m�j�b�a�`�a�a�a�a�a�a�a�a�Z�������������������r�g�Z�F�8�7�.�5�F�Z������"�;�v���x�a�C�/����������������价�л��� �5�I�L�A�4�'���û������������������мּ�ּ�����������r�g�V�P�Y�r��ÇÓàììù��ùùìàÓÇ�~�z�z�z�|ÄÇ�0�<�I�O�R�I�<�5�0�)�0�0�0�0�0�0�0�0�0�0���������������������������������������ſ��������(�,�(�������ݿ׿ѿݿ��5�A�N�Z�g�h�l�g�g�[�Z�X�N�H�A�=�5�2�5�5��'�3�@�B�@�3�3�'��������������������������������������� �����޹�����������ܹϹ������������������ù���������*�5�B�6�3�*��������������뽒�����������������y�l�^�W�?�G�S�U�`�����4�M�Z�s��������~�s�f�T�D�8�)�(��� �4����������������~�z�����������������������������������������ùìåàì�����������������������������������������������	�����������������������������	���!���	������������	�	�	�	�	�	�N�[�g�t�u�}�z�|�t�g�[�V�N�F�B�@�B�D�N�N�������
����
������������������������ŔŠŭŮŲŭũŠşŔōŌŔŔŔŔŔŔŔŔE+E7ECEPE\E]E^E\EPECE7E6E*E+E+E+E+E+E+E+ Y M E -  G . E n  < 6 J M p S @ L ' e @ 3 v 7 7  ] . B : / Y 3  S ` Y : : $ ; ` [ H h ' 3 � ; + i 9 \    �  �  t  B  �  �  ~  0  �    �  C  �  �  �  �  �    �  �  D  �  �  �  c  M  %  �  }  �  p  !    O  4  �  Q  �  K  �    �  �  �  g  �  �  x  �  =  �  �  ����#�
�u���
<��=L��<49X<D��<�o>�%=P�`<��
=���=�hs=8Q�=��<���=+='�=��<�`B=��<�/=D��=�9X=L��=#�
>7K�=#�
=aG�=�^5=�x�=��`=q��=,1=@�=��=y�#=aG�=���>\)=�O�=���=��w=q��=���=�O�=��=�O�=�;d=� �=ě�>oBB�&B5�B
�lB�xBQ�B˄B$�BOB	(�B �A���B^�B"��Bz6Bq�B�B
~lB��Bw�B!yB!�BV�BY�BY9B�B��B%;B ��B	��B��B�B�9B!��BqQB�LB�B:�B�iB0�Bw�B�CB,�FB�5B#�4B~�B?B)z)B#�FB[&B��A��nBcB=(B��BD�B#BˉB@BB$��B?�B	?�B!?�A���B4�B"TpB?�B@�B�=B
,�B{�BT�B!@-BñB<�BD]BC�B1aB�B>�B ��B
�B��B>�BF?B"2,B��B7�B�KB=�B�UB�BV�B�:B,��B��B#q�BGB�_B)��B#�#B?�BʴA�zWB=�A�;hA�M�?��2A�g�A��A�|A�[�@�L�Aj�A��I@�j�A�8A�:�A7�5AGA�B�A��A�)8A��/=���?���Au�jA�)@A`vA'�7AkV!A�V�C��wA���A���A��X@��&@��9A�,�A�A���A��A���?��fA�t�>lx�A�Z2AA?�$@��;A��A��H@\��AZu�A��"A�vpA�NiC��A�z�A��?�=_A�P�A���A��wA�@�EAi+Aԅ6@�?A�lUAě6A7�AH�BvPA���A��A�q�=|�H?��Au3A�v�Aa-A(��Aj�GA��
C��A��	A���A�n@�1x@��#A��A�qA�PPA�^%A�?�V�A���>{�A��A��AA�@���A��IA�|@\�JAY�YA���A�n�A�t�C���         
         1   
        m   %      ?   7      /               	            @         �   	      ;   Q   E         	         
      ]      2         +            !   	   
                     9            1   #      '   #   )   1                           %         !      !   5   3   #                        !      -   #                                                            #            %   /                                          !   -   3                                 )                              N�k�Nm�NL��N/�=OC�OQ<wNz��N"�N��O��O�6�N*�6O�C�O\m�O���P;�:N���Os6�O_nND@wN,׈O>.Nj��OtO,�*N��OR�COaANXSMO�_qPCw�PLO�eOT�N�O�N�|�N�\�N1&wO-{O�\�O2W|O��:O��.N$O��~N�KAN�N�uN��N�+N��Na��  t  �  *    �  �    a  �  �  N  t  �    �  �  �  x  �  {  S  �     A    ~  �    �  r  �  	j  	  c  3  +  �    �  .  \  V  C  �  �  8  p  t  C    �  �  ��49X�49X�ě���j<t�<�h;o<t�<D��><j<�t�<�o=,1=+<��
<�/<���<��
<�9X<�/<��
<�1<�j<�h=ix�=o<�`B=��T=o=o=0 �=\)=e`B=�P=��=�w=<j=8Q�=<j=L��=u=L��=]/=ix�=]/=u=m�h=}�=�o=�1=��w=�-=����������������������������������������cdhnt����thcccccccc��������������������,)(/8<BHPUZ_YUH<0/,,����������������������������������������#''#")5>BLNPONB5)""""""=<=@HN[gt�����tg[NC=��������������������248;>HJLH;2222222222�������
	��������������������������?=@GKO[hjsvyxthc[RD?	")N[gij_WNG5(LNNRY[ginqnkg][NLLLLY\kt������������tf[Y���
#/37<9/#
����������������������������������������������)4520-)%�%%#&)5BNONB?5,))%%%%�����������{xx{���������������{���
#'%#!
 ���,+-/<HMUansznaUIH</,��������

�������������������������MYcnt����������t[TNM�������)6;:6)��������)BO[a`WJ8)���NJFFDFO[ht}���|sh[ON��������������������LJLN[^^[PNLLLLLLLLLL������������������ywxz|�����������~zyy��������������������))0*)����������������������������������������������������������������������������������������������������


������������}}�����������������),-+)	|�����������||||||||����

����������
#$///.&#
��������������������UV`amxzzzrmaUUUUUUUU��������������������ĦĳĿ������������ĿĳĦğĞĦĦĦĦĦĦ������
���
��������������������������L�Y�]�e�l�g�e�Y�N�L�F�K�L�L�L�L�L�L�L�L��������������������������������������������������
��������������������A�N�Z�g�s�}�������y�g�Z�N�A�5�(�&�)�4�A�#�/�<�>�C�<�9�/�#���!�#�#�#�#�#�#�#�#�ܻ����������ܻ׻ܻܻܻܻܻܻܻܻܻܿ`�m�y�q�m�g�m�m�m�`�^�[�U�Y�`�`�`�`�`�`����)�6�=�D�F�C�;�)������������������'�4�@�M�P�>�>�4������޻ػ߻������(�*�(� ���������������<�H�U�a�j�j�a�H�<�/�#���
����#�/�<���(�4�A�M�X�Z�b�X�M�4�(����������������ʾ׾۾߾Ⱦ������f�Z�D�5�A�N�f���ƎƧ���������������ƳƁ�u�e�h�uƆƎ�)�5�7�B�N�U�T�N�B�5�)������)�)�)�)��"�/�3�7�=�<�;�5�/�"�� ��������	����(�,�5�:�A�D�D�5�������������������ùŹϹ˹ù���������������������������'�3�7�6�3�'����������������������Ŀ˿пѿֿѿĿ������������������m�z�����������������������z�m�l�m�m�m�m��"�.�;�G�J�Q�M�G�;�.�"�!��	���	�����Ľнݽ�����ݽؽнĽ��������������m�y�|�����������y�m�g�`�T�P�R�T�`�e�m�m�����������������������s�g�e�f�s�u�����D�D�D�D�D�D�D�D�D�D�D�D�D�D{DqDhDlDoD{D��a�m�y�z�����z�m�j�b�a�`�a�a�a�a�a�a�a�a�Z�������������������r�g�Z�F�8�7�.�5�F�Z�����"�;�H�\�e�e�\�K�5�"���������������𻷻л����5�H�K�?�4�'���û������������r������������Ǽü�����������r�j�h�o�ràêìùú��ùøìàÓÇ��{�~ÇÓÓàà�0�<�I�O�R�I�<�5�0�)�0�0�0�0�0�0�0�0�0�0���������������������������������������ſ��������(�������ݿؿҿݿ����5�A�N�Z�g�h�l�g�g�[�Z�X�N�H�A�=�5�2�5�5��'�3�@�B�@�3�3�'���������������������	�	�����������������������޹ùܹ���������޹Ϲù�������������������������*�5�B�6�3�*��������������뽒�����������������y�l�_�Y�G�G�S�[�`�����4�A�M�s�����������w�s�f�[�L�@�1�.�.�4����������������~�z������������������������������
���������ùíæâìò�����������������������������������������������	�����������������������������	���!���	������������	�	�	�	�	�	�N�[�g�o�t�x�t�t�h�g�g�[�N�J�C�H�N�N�N�N�������
����
������������������������ŔŠŭŮŲŭũŠşŔōŌŔŔŔŔŔŔŔŔE+E7ECEPE\E]E^E\EPECE7E6E*E+E+E+E+E+E+E+ Y M E - ' C . E n 	 ? 6 G H l Q @ L   b @ 3 v . .  d  B :  Y   S ` P : :  * ` X 6 h & 3 � ;  i 9 \    �  �  t  B  -  �  ~  0  �  �  �  C  (  �  (  �  �    �  �  D  �  �  K  e  �  �  �  }  �  0    �  :  4  �  &  �  K  R  s  �  (  ^  g  z  �  x  �  �  �  �  �  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  t  t  s  s  o  k  g  b  \  W  P  J  C  <  6  /  ,  5  ?  I  �  �  �  �  y  r  i  `  W  N  B  4  %    	   �   �   �   �   �  *  =  O  X  _  ^  Z  S  J  >  0    �  �  �  �  f  =    �          
  
  	                    !  *  4  =  :  u  �  �  �  �  �  �  �  �  ~  W  &  �  �  U    �  R  h    #  *  =  [  s  �  {  w  �  �  �  �  �  �  o  )  �  H   r    �  �  �  �  �  �  �  f  B    �  �  �  C  �  �  l  $   �  a  ^  \  Y  W  T  R  P  N  L  K  I  H  F  E  C  A  @  >  =  �  �  �  {  t  m  h  b  \  V  O  F  =  5  ,       �   �   �  �  y  V  �  �  �  �  L  �  �  �  �    �  f  1  X  �  �  �  %  F  L  =  %  
  �  �  �  �  c  6  +  �  �  �  U  �  �  �  t  j  `  U  K  @  3  &      �  �  �  �  �  q  S  5     �  X  �    e  �  �  �  �  �  �  �  `  �  ^  �    �  v  �  �  y  �  �  �  �          �  �  �  6  �  n    �    �  �  �  �  �  w  e  J  &  �  �  �  �  m  J    �  �  7  �  !   m  �  �  �  �  �  �  �  �  q  B    �  �  ,  �  �  2  �  �    �  �  �  �  �  �  �  �  �    s  j  a  W  J  =  -  
  �  �  x  l  [  D  +    �  �  �  �  �  �  {  ]  A     �  �  �  j  �  �  �  �  �  �  n  T  6    �  �  �  j  ?  	  �  |  *  �  �  6  ;  <  :  @  a  w  ]  d  c  R  ]  L  ,  �  �  l  �    S  X  \  ]  [  X  P  H  %  �  �  �  ^  '  �  �  }  C     �  �  �  �  �  �  �  u  h  W  @  %    �  �  �  o  L  $  �  �           �  �  �  �  �  �  �  �  �  �      �  �  �  �    /  <  ?  A  ;  2  ,  %      �  �  �  j  2  �  �  �  =  �      8  [  �  �  �  �        �  �  L  �  K  �  �  j  Z  k  w  |  ~  z  o  Y  ;    �  �  �  r  C    �  �  w  ;  �  �  �  �  �  u  e  T  G  F  i  |  �  �  �  q  T  ,  �  `  /     �  <  �  �      �  �  #  �  �  �    �  %  V  �  ~  �  �  �  �  �  �  �  �  �  o  X  ?    �  v  >    �  �  |  r  `  I  .    �  �  �  Z  +     �  �  �  �  �  x  P    �  _  }  �  �  �  �  �  �  �  v  W  1    �  �  a  �  L  .    	h  	h  	R  	&  �  z    �  �  �  �  n  3  �  x  �  3  �  <    �  �  �  �  	  	  	  	  	  �  �  N  �  e  �    @  b  q  �  W  _  I  /    �  �  �  Q    �  �  �  l  P  0    �  �  �  3  3  4  4  4  3  0  -  *  &  %  $  $  #  #  K  �  �  �    +         �  �  �  �  �  �  s  ^  H  3      �  �  B  �  �  �  �  �  �  �  i  G    �  �  �  J  �  �  �  K  m  �   �      �  �  �  �  �  �  u  \  D  +    �  �  �  �  �  �  �  �  �  �  �  �  �  z  ]  =    �  �  �  T    �  �  F   �   �  '  #  .  ,  ,  '  "         �  �  �  e  '  �  �  Q    �  �    Z  V  >    �  �  Y    �  3  
�  
N  	�  	3  i  j      V  M  B  0    	  �  �  �  P    �  �  [    �  �  h  3    5  A  �  �  �    %  /  8  =  9  *  	  �  �  0  �  9  �  e  �  �  �  �  �  �  �  �  �  �  r  I    �  �  q    �    u  �  v  f  V  F  3       �  �  �  �  �  �  �  w  F    �  �  8  8  4  '    �  �  �  e  $  �  ~  )  �  �  5  �  a  �  A  p  `  P  C  >  ?  E  K  A  .    �  �  �  �  _  7    �  �  t  a  O  <  )      �  �  �  �  �  �  �  �  Z  1  	   �   �  C  5  '      �  �  �  �  �  �  �  �  �  �  x  k  ^  R  E  �  �  �        �  �  �  ^    �  {    �  �  b  �    �  �  �  �  �  �  q  ]  H  4    
  �  �  �  �  �  �  �  }  l  �  �  |  _  B  &    �  �  �  �  �  y  a  C    �  �  D    �  �  ^  3    �  �  �  x  _  *  �  �  �  s  S  .    >  ?