CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?ɺ^5?|�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�5�   max       P�]�      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��9X   max       =��T      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F�z�G�     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p�   max       @vu�����     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P            h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�p�          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >^5?      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�и   max       B2�      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�{�   max       B3;�      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C�m�      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Є�   max       C�^�      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          I      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�5�   max       P��      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?�"h	ԕ      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��9X   max       >	7L      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @F�z�G�     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�����̀   max       @vu�����     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P            h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @��          �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���%��2   max       ?� ě��T     �  K�                  3   .      	   �         E   =         u   V      @      /            	         ?   
   	   !   c         /   o      6                     4      �         cN��O�mN�pOÜOiqOѢ�O��1N�GNG,'Pb8uO��*Oh��P:W�O���O��3N��P%��P�]�N��RP��N���P(�:N�$�N�ѳN0��N�D�O`��N4��O�.N�GDN�m�O�=P��IOz��OJ�O��O��QNg�zO��RNN��M�5�NiJpO!@%O%k�P(��O��pO��SO�`vM�L�N��HO�#㼴9X�o�ě�;D��;�o;��
;ě�;ě�;�`B<t�<49X<D��<D��<T��<�o<�C�<�C�<�1<�1<�9X<�9X<�j<�j<���<���<���<���<�h<�h<�h<�<��=+=�P=#�
='�=0 �=49X=8Q�=@�=@�=@�=L��=L��=]/=]/=q��=}�=��=��=��TQOUYahmnowpnaUQQQQQQTSOMNQUbbnyyyunmhbUT����������������������������������������4667:<<HJUVZ`YUSH<44����6;?C@6)��TY^gn���������zna^VT��������������������Zanz~���zngaZZZZZZZZ>>I[g�����������g[P>�������
#$#
���'*/5BNQ[cgkjg[NB4/)'celnt�������������jc����������������������������������������"*/0;<<;3/"fcgn�������������rnf�����AU^cog[B,������������������������)5BEJJGFB5)��~y��������������~~~~��������
$''%�����[[`grt��������tmh_[[[gkt����������tsg[[456BDIOOPOBB76444444������


	������YUUV\ahu��������uh\Y�����������������������
#/AEJLJNH<7#
�,%'*.0<DFDA<:0,,,,,,��������������������������
#<DLP</
�����������)2;8)����C><<@DHTmyz�zwlaTRHCmjqt����������utmmmmUNHA/#�������
#/BHU��������������������iahlnz�~zqniiiiiiii�}{}������������������������������������413568=BDCB644444444>;BOZ[hmnh[OOB>>>>>>MGEN[gltx����}tgc[NM��������������������stz��������������~}s�������������������������.23/(����������
������������������),66623)��������
 $$
������������������ �������_�l�x�������������������x�l�a�_�S�L�^�_�����������������������������������������ûƻлܻջлȻû������������������������H�U�a�n�zÇÇËÇ�|�z�n�a�U�J�H�?�C�H�H���������Ⱦ���������s�Z�M�;�6�8�>�M���E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E�E�Eͼ����ʼּ�ּּ̼ʼ����������������������<�E�A�=�<�4�/�)�'�+�/�3�<�<�<�<�<�<�<�<��6�O�e�n�t�o�h�[�>�)�����������������/�;�H�T�a�m�p�j�a�T�;�"���
���������������������������������������������������'�6�B�D�@�)�"�����������������������������ìÚÓÇ�~�{ÀÐàêù�������(�A�M�Z�e�l�m�c�Z�M�A�4�(�������T�a�m�m�y�z���z�q�m�a�\�T�S�N�R�T�T�T�T�ûܻ�� ������ܻ�������������������Ƴ�������8�7�����Ƨ�u�@� ��O�f�uƃƳ��"�.�2�5�4�.�"��	��	�	��������H�P�a�t�ÆÊÇ�z�a�/�������	��#�;�H�/�<�?�H�K�H�D�<�/�/�/�#���#�,�/�/�/�/������B�N�g�z�w�b�W�B�5�������������Ѻ@�@�L�Y�e�g�l�e�_�Y�S�L�@�>�8�3�1�3�>�@��"�(�/�5�5�5�1�(�&�����������������������������������������������������������Ⱦž���������������y����������ʾ׾���	����������ܾʾƾþǾ�²¿����¿²¦¦­²²²²²²²²²²���ʾھ����ʾ�������f�_�a�f�n�x�����M�Y�f�r�������r�f�Y�O�M�G�M�M�M�M�M�M���������������������~�z�v�x�z�����������������)�-������������������ú������0�nŇŔŝŘō�n�P�0����������������Ŀ������������
�������������ľĳĳĮĿ�����������������������������������������m�`�Y�T�N�E�;�/�:�G�T�m�y���������y�q�m�ʼּ�����������ּʼ������������ʺ@�L�Y�e�i�n�e�Y�L�C�@�?�@�@�@�@�@�@�@�@������� � ������������þöû��������4�5�A�G�C�A�4�(�$��(�,�4�4�4�4�4�4�4�4�����!�$�!��������������������������-�:�?�@�F�G�L�F�:�5�/�-�-�)�-�-�-�-�-�-�m�y�������������������y�i�`�X�]�`�f�l�m�`�e�l�y�����������������y�s�l�g�c�`�U�`�������������������������g�W�Q�Q�_�s�����;�T�m�y�����������z�a�H�+������	�"�/�;���������ÿѿٿڿѿο�����������}������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DwD~D�D��ɺֺ���غֺɺ������źɺɺɺɺɺɺɺɺ��������������������������������������EiEuE�E�E�E�E�E�E�E�E�E�E�E�EuEnEhEgEgEi u ? G 6 A X 6 5 } # 3 , F A # 1  M + I . ` Q X R ] B p J C ` X ? 4 4 K 1 = >  l � < 6 ( _  / o 3     k  f  �  A  8  '  m    �  �    �  S  �  9  �  �  $  �  �  �  x    0  ]  �  �  8  p  �  
  a  �  �  =  �    �  �  [  7  �  ]  r  �  �     �  ,  �  #���
��o<o<��
<#�
=aG�=P�`<�t�<u>V<���<�`B=��
=��P=49X=\)>	7L=�"�<���=� �=t�=�\)=8Q�<�`B<�h=C�=#�
=#�
=�j=��=��=�o>+=m�h=Y�=�->�=P�`=���=Y�=P�`=T��=��=�O�=�1=��=�Q�>^5?=�C�=���>7K�BDB'�^B"'�B"�QB&�B�BZ�B!��B�sB	»ByiB�B
��B!�dB  fA�иB�B�7BՌB�"Bk�BշBUfB
1BDB#�B2�B'�B��B%�<B��BX�B�A���B1BbUBs6B��B��BY�B��B��B	T�B,�B��B1(B�B-B�ZB�BgKB>�B'��B"7�B"�B:}B�0BuB!�UB�"B	��B�qB�OB
�[B!�.B��A�{�B?mB?nBB��B�B>�B}�B/�B@ B#��B3;�B�UB��B%�vB �BJ�B�A�x�B��B��BHaBDB��B� B��B�B	L
B,� B<�B�B�/B8�B�B�B��A�J�@�?t@�x@�T�Aƿ1AB�hC�m�@��uA�A�V�A�,yA���A��A�t{A9��A��@��B��A^k�AôJA��|A��?���A�_)@� BAJ��AU�zA��XAL�@�A�ğA�sA�A��As̗Ah�Aي?�`A�f�A8ѝ@^�|@{cOAmIA.�A���A�޺At�C��@9U@��C�	�A��9@�4@�0@�T�A���AAI�C�^�@���AØ�A�x�A��A��A�xoA��A:�A���@� �B�A]��A�}sALA�|,?Є�A�x�@��1AK/?AW�A�}�AL�@ݦ�A��QA҂^A�}uA�bWAs��Ak&A%�?ؾ�AсA9 �@\�@{��Al�A��A���A�~QAsGC��@>�h@�C�y                  3   .      	   �         E   >         v   W      A      0            	         @   
   	   "   d         /   o      7                     5       �         c                  %            /         -            '   I      '      /                     %         +   7             !                        )   '                                                                     I            '                                 /                                     )                  N��O�mNa�OÜOiqO�O��1N�f�N#��O���O��*Oh��O�aAO/׿N�N���OwM�P��N��RO�a�N�I�P�N�l�N�ѳN0��N�D�O`��N4��Ox�]N�GDNyx�O�{�P[��O7X2OJ�O��Oi�vNg�zOWt�NN��M�5�NiJpN���N���P(��O��OKŴO e�M�L�N��HOlIn  |  �  j    A  �  �  K  5  �  �     �    �  �  �  �  �  V  �    �  �  o  �  �  �  9  �  9  &  	�  �  �    y  �  �    }  �  �    �  {  �  #  c  p  ���9X�o�D��;D��;�o<�C�;ě�<t�<o=���<49X<D��=#�
=C�<�<�t�=�7L<�9X<�1=#�
<�j<�<���<���<���<���<���<�h=P�`<�h<��=��=L��='�=#�
='�=�1=49X=y�#=@�=@�=@�=aG�=ix�=]/=�7L=��>	7L=��=��=�jQOUYahmnowpnaUQQQQQQTSOMNQUbbnyyyunmhbUT����������������������������������������4667:<<HJUVZ`YUSH<44��)49;<96-)
TY^gn���������zna^VT��������������������[anz{���znia[[[[[[[[TRTW`gt���������tg[T�������
#$#
���'*/5BNQ[cgkjg[NB4/)'ooqu{�������������to����������������������������������������"/;<<;//"�������������������������@T]cnm[B-�����������������������	
")5>ACB@>5)������������������������������ $$ 
����d^bhist��������tjhdd[gkt����������tsg[[456BDIOOPOBB76444444������


	������YUUV\ahu��������uh\Y��������������������
#/5<?@@</#,%'*.0<DFDA<:0,,,,,,�������������������������#/<?HL</#
����������$,12)"����C@ADHJTafmtytmha\THCmjqt����������utmmmmUNHA/#�������
#/BHU��������������������iahlnz�~zqniiiiiiii����������������������������������������413568=BDCB644444444>;BOZ[hmnh[OOB>>>>>>YOS[gt|���xtg[YYYYYY��������������������stz��������������~}s������������������������ &)+,(���������

��������������������),66623)������
"!
�������������������� �������_�l�x�������������������x�l�a�_�S�L�^�_�����������������������������������������ûƻлܻջлȻû������������������������H�U�a�n�zÇÇËÇ�|�z�n�a�U�J�H�?�C�H�H���������������s�f�Z�M�H�A�?�?�F�M�Z�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E�E�Eͼ����ʼּڼּѼʼǼ����������������������<�D�@�<�<�2�/�,�(�,�/�4�<�<�<�<�<�<�<�<���6�B�O�T�X�U�M�B�6�)�������������/�;�H�T�a�m�p�j�a�T�;�"���
����������������������������������������������������	��� ��	��������������������ìù������������ùìàÓÍÇÅÇÌ×àì�A�H�M�W�T�M�A�?�4�-�(���(�4�7�A�A�A�A�T�a�l�m�x�y�o�m�a�_�T�T�O�S�T�T�T�T�T�T���ûлܻ����������ܻлû�������������Ƴ�����
��2�6�����Ƨ�u�\�#��O�g�vƅƳ��"�.�2�5�4�.�"��	��	�	��������/�<�H�U�a�g�n�t�x�n�a�U�<�/����� �/�/�<�<�H�J�H�C�<�/�#���#�.�/�/�/�/�/�/��������)�B�[�c�g�c�P�B�5�����������ź3�@�L�V�Y�e�f�j�e�[�Y�W�L�@�?�:�3�2�3�3��"�(�/�5�5�5�1�(�&�����������������������������������������������������������Ⱦž���������������y����������ʾ׾���	����������ܾʾƾþǾ�²¿����¿²¦¦­²²²²²²²²²²�������ʾӾ׾ݾ۾Ҿʾ��������������������M�Y�f�r�������r�f�Y�O�M�G�M�M�M�M�M�M�������������������z�w�y�z�����������������������������������������������0�I�bŇőœō�{�I�0����������������
�0�������������� ����������������Ŀĸľ�̿����������������������������������������m�`�Y�T�N�E�;�/�:�G�T�m�y���������y�q�m�ʼּ������� ��������ּʼ��������ʺ@�L�Y�e�i�n�e�Y�L�C�@�?�@�@�@�@�@�@�@�@���������������������������������4�5�A�G�C�A�4�(�$��(�,�4�4�4�4�4�4�4�4�����!�$�!��������������������������-�:�?�@�F�G�L�F�:�5�/�-�-�)�-�-�-�-�-�-�m�y�����������{�y�x�m�e�h�l�m�m�m�m�m�m�����������������y�m�l�k�j�l�y�{���������������������������������g�W�Q�Q�_�s�����;�T�a�m�r���������z�m�T�H�?�/�*�#�,�;���������Ŀɿ̿ÿ�����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ɺֺ���غֺɺ������źɺɺɺɺɺɺɺɺ��������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�EuEqEkEkEkEuEwE� u ? 3 6 A F 6 2   3 , A 8  4  K + C ( ^ R X R ] B p . C H M @ ' 4 K & = *  l � = : ( E   o 3     k  f  h  A  8  )  m  �  �  P    �  F  �  �  �  �  �  �  O  �  �  �  0  ]  �  �  8  �  �  �  f  �  {  =  �  �  �  �  [  7  �  �  �  �  %  �    ,  �  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  |  {  {  {  z  z  z  y  y  y  z  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  g  Z  L  A  7  .  S  [  c  i  j  c  V  D  /    �  �  �  �  �  p  a  ]  f  �      �  �  �  �  �  �  y  R  (  �  �  �  e    �  f      A  ?  =  <  4  +  #      �  �  �  �  �  �  m  T  ;  #      A  [  s  �  �  �  i  d  C    �  �  �  c    �  �  �  \  �  �  m  G  +    �  �  �  �  c  2  �  �  I  �  K  �  :  �  ;  B  G  J  K  J  C  8  )    �  �  �  �    Y  2    
  $  #  +  2  9  @  G  O  V  ^  e  }  �  �  �  �  x  ]  @  #      �  �  �  p  �  u  �  �  �  �  �  >  �  �  �  W  t  �  T  �  �  �  �  y  s  l  d  U  >  !    �  �  �  �  �  �  �  V       �  �  �  �  �  �  �  y  l  [  D  (  	  �  �  �  �  �  �  W  �  �    ;  e  �  �  �  r  N    �  C  �  <  �  �  M  =  �  �    D  b  y  }  r  _  =    �       �  &  �  �  �    L  w  �  �  �  �  �  �  �  �  �  �  �  }  ]  :    �  �  r  �  �  �  �  |  e  J  -    �  �  �  �  g  *  �  �  P    <  	/  	�  
�    d  �  �  �  �  �  �  2  
�  
  	D  [  "  �    �  �  �  m  R  2  	  �  �  �  �  �  �  �  �  .  �    �  �  �  �  �  �  �  �  �  �  �  y  r  l  f  `  Z  W  T  Q  N  K  X  �  �  *  J  U  Q  6    �  �  �  L    �        �   �  �  �  �  �  �  �  �  �  �  d  C  !  �  �  �  |  B  �  �  H  �  �  �    �  �  �  |  �  �  y  A    �  v    �  �    T  �  �  �  �  �  �  m  V  ]  �  q  N  )    �  �  �  J    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      o  o  n  n  n  m  k  j  h  g  f  h  i  k  l  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  x  o  f  ]  R  H  ;  .      �  �  �  �  �  �  ~  f  H  *    �  �  �  �  �  }  V     �   �  �  �  �  �  �  �  �  �  �  �    )    	  �  �  �  �  �  �  C  �  �    $  /  6  8  .    �  �  l    �  2  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  i  W  /    �  �      2  3  )      �  �  �  �  �  �  �  w  Z  ;  !    �  �  �  �  �  &      �  �  =  �  x  <  �  }    �  F  �  |  	u  	�  	�  	�  	�  	�  	�  	V  	   �  �  }  <  �  n  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  E    �  �  ]    �  �  �  �  �  �  �  �  �  �  �  s  ^  C  #  �  �  �  �  \  1  �  �    	  �  �  �  �  x  \  I  6    �  �  j    �    c  �     
�  �  v  �  2  W  n  y  s  U    �  *  �    
+  	!  �  �  �  �  }  o  b  U  I  <  0  #      �  �  �  �  �  z  R  '  �  $  U  g  r  �  �  �  �  u  B  �  �  U  �  |  �  )  !  �  �              �  �  �  �  �  �  �  ~  n  ^  \  �  �    }  r  g  \  Q  D  /      �  �  �  �  �  i  K  -     �   �  �  �  �  �  �  �  �  w  k  ^  R  F  :  .  "    �  �  �  ~  �  �  �  �  �  �  �  �  �  �  �  q  Y  @  &  �  �  �  �  �  �  �  �  �  �        	  �  �  �  �  g  ;    �  �  o  V  �  �  �    d  K  /       �  �  �  �  K  �  �  6      �  �    <  Z  o  x  h  Y  C  &  �  �  �  L  �  s  �  Q  �  �  �  �  �  �  �  �  �  �  �  �  �  W    �  q    �  #  �  ~  .    �  �  =  �  �  �    !    �  "  I  H  4  �  j  x  
�  c  _  [  X  T  P  M  D  :  /  %      �  �  �  |  Q  '  �  p  k  f  f  e  ^  U  I  >  2  &      �  �  �  �  �  n    J  �  �  �  �  {  D  �  �  I  �  F  �  �  �  �  
^  	  �  �