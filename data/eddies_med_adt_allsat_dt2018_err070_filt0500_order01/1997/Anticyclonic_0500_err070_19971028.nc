CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ł$�/      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Q   max       P�t�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��P   max       =� �      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F5\(�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @vs��Q�     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3�        max       @Q            p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�P�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       >:^5      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��q   max       B0'�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B08�      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?&   max       C�z�      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?0q   max       C�hI      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Q   max       P(B�      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����#�   max       ?�	� ѷ      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��P   max       =Ƨ�      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F5\(�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ۅ�Q�    max       @vs��Q�     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q            p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�9�          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D;   max         D;      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�	� ѷ     0  O�      
      "      �         e         "   F   
   �   '               :   $                  (      *                  #                  4   =                  	   S      '   A      NmP�OFNHO��N�7O��XOU�O�
P�vNO�N���Pf�P���OI�VP�t�O��bO�jeO�QN_-SOx�;O׆!O�yOd.}M�QOl�N,��O���P(B�N5�P�O)SvO��OeE�Nك�O���OЙ�N�^WNH�N�βN�N��O��\P�]O�2N7��O��N xtN� =N{s�P#'�O	�BO(�.O���N>�N�����P��h�e`B�D���ě�:�o;o;��
;��
;��
<o<#�
<T��<T��<e`B<e`B<�o<�C�<�t�<��
<ě�<�<��=+=\)=t�=��=�w=�w=#�
=,1=,1=,1=,1=0 �=8Q�=@�=D��=H�9=P�`=T��=T��=Y�=e`B=m�h=y�#=�o=��=�7L=�C�=�t�=���=��
=���=� ����������������#0<EIJI<70+#�������������������� #/<HU`mlaUH</#)6BOXVTOFB@61)$�������
&)'!�����tvwy���������������tz}�����������������z�����7EGGB5)��������������������������������������������^`gsy�������������g^����)BS[[VH=���������� " ���������5BPZR[musc5���DEHN[ht�������wth[OD�����1@JJLHA5)��	"-/;HOQHE;/%"	YX[bgstzwtg[YYYYYYYY�����������|y}����������������|013:<?>HUZintsmcUH<0��������#
��������������������������*6COWZZXOC6*)6BFB?6))���������������������������
!*>@9(#
��


�����5DFE5)�����feeehmz���������zmff�������������������������� ���������������������������������
#!
�����������������������������������������������������������TMNUbcnqtsnbbUTTTTTT���"���������������zuyz�������������������
#%!
���������)BOdpnh[O)�	
#$-,,*'#
htutqh[WPX[hhhhhhhhhV`t��������������t[Vmmnvz|{zynmmmmmmmm��������������������)6:63,) mlhhbn����������zm�~������������������������

����}{{����������������}��

���������)15652)�Ľнݽ��ݽнĽ½��ĽĽĽĽĽĽĽĽĽĻ������������������������x�l�e�l�p�x������������������������������������������������������������������������޼ּټ�߼��ּʼ��������������ǼʼҼּ�D�D�D�D�D�D�D�D�D�D�D�D�D�D}DoDkDmD{D�Dƹܹ��������������ܹ׹ϹʹϹйѹ�������������������������ĴĳĨĪĳ�������<�b�u�y�w�n�j�b�U�I�
�������������
�#�<E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�Eٽ�����������ݽнҽݽ�������A�Z�_�s�����������f�M�4�!�(�0�+�3�3�;�AƳ�������	�����ƳƎ�u�h�6�/�C�\�zƔƳ������������"�'����������������������6�O�jĬīĚā�[�A�)�������������A�M�Z�f�w����������s�Z�A�1�����+�A���ݿ��������ݿѿĿ������������������T�a�l�r�z�{�~�}�z�m�g�a�T�L�H�F�G�H�P�T�y���������������y�v�r�w�y�y�y�y�y�y�y�y�;�G�T�`�m�x�z�m�T�G�;�.�"�����"�*�;�6�B�O�V�X�X�Q�S�O�6�)���������"�)�6�T�a�z�����������������������m�a�W�R�Q�T�����������������������y�t�q�l�r�x�y��A�N�R�Z�Z�\�Z�Z�N�I�A�@�A�A�A�A�A�A�A�A�`�m�y�������������y�m�`�U�M�I�H�K�T�X�`�'�'�-�3�/�'������'�'�'�'�'�'�'�'�'�������������������������������s�g�g�s����"�;�`�y�������y�m�T�;�.���߾������������ʾҾʾ��������������������������#�/�;�8�7�/�*�"��
�����������������
�#ŠŭŹ����������������ŹŵŭŠŜŕŕŠŠ�Ŀѿݿ�����(�4�*�/�(�����Ŀ����������������������������������z������������Óàìõù��ùùìàÓÇ�~�~ÇÎÓÓÓÓ�)�N�[�d�]�N�I�6�)�����������������)���������������������������v�q�s��������-�3�:�F�R�S�T�S�F�:�-�$�!��!�"�-�-�-�-��������s�f�d�b�f�s�|����������������������������������������f�p�j�j�i�f�`�Z�P�O�O�V�Z�b�f�f�f�f�f�f�
�	�������
������!��
�
�
�
�
�����	��"�/�4�C�F�G�;�/�"��	����������ûܻ�����������ܻл˻ʻλ������þ(�4�A�M�Z�^�r�s���s�Z�M�A�4�,�(� �$�(��߿ݿؿѿοοѿݿ������������������������������������x�t�{���������������������������������������������������������������������������������z�y�t�y�����������������������������������������������ɺֺ���غԺɺ�������������x�|���(�5�A�N�Z�[�Z�X�J�A�5�.�(�%�����"�(E�E�E�E�E�E�E�E�E�E�E�E�E�EuEtEiEmEuExE���'�@�M�V�Y�Y�L�4�'���������������C�D�O�\�`�\�[�O�I�C�B�B�C�C�C�C�C�C�C�C�g�t�w�}�t�k�g�f�e�g�g�g�g ? ' ^ : X 4 , < > [ U - 2 T = : . a ' ; M D E r 7 n < 7 g = ) X L 3 | Y > u A � � ? Q Y R n Z C 6 e A 8 . Y `    �  <  P    <  `  �    [  �  �  �    �  v  �    �  s  �  7  �  �  L  �  �  �  �  �  �  g  �    �  �  ,  �  �  �  ;  ^  �  �  o  H     >  �  �  �  C  k  �  B  ������49X<�9X;�`B>�<�/<���=�
=<T��<#�
=0 �=��<�j>:^5=Y�=\)<��<ě�=49X=��=��=ix�=\)=Y�=0 �=m�h=��w=8Q�=��=aG�=e`B=�7L=ix�=�+=��
=}�=]/=aG�=ix�=m�h=��=�l�=��-=�%=�1=�O�=��-=���>��=� �=�x�>z�=�9X=ȴ9By�B%��B�eB�B��BP�BnB��B�BkB��Bn�B��B@B�B��B1�A��qB	H�B��B?�B��B#M�B8�B0'�B��B.*B�WB&�BpZA���B��B�B!�+B��B�B PBB'��B�B|�B�VB%�B$��B��BDqB:�B,t7B�B��BLkB��B��B�BBK#Bu�B%��B�CB��B8B>]BBwB`B6UB@�B  BN�B�'B�{BB#B�BB8A�B	y�BÚB��B�B#B_B �B08�B��B0�BS"Bf~B�ZA��B��B<|B">�B��BűB H�B;'B'��BY%B��B�B��B$��B��B�B@iB,HbB>�B��B~�B��B�B��B�A)9@��>B�(A�`�@�߲C��?&A�Y\A�/C�z�A-m�AA�eBx4A���A�=$A=��Ay�uA���AnwDAd�JA�e?A��)@�>�A�`2Aj��?��A��$Ab#[AMXOA�g�A�v�A���A��SA�j�A���A��@z��AD�A9A@{A�PA��@�8^A=G�A|i�A�#�@���A�@ա@$DlA���C��@�x�B3bA�͵A(Ӟ@��B�OA�~g@��C��?0qA��A�KC�hIA+�AB��BêA�=�AӁA>��Av��A�}�An"�Ac��A�2A��?@��A��Aj��?���A�#�Aat�AMzA��rA�BAA�yA�a�A�m[A���A��@|n�AC,�AgA?WA���A��I@�n�A;\EA}A��6@��6A @��@#��A��CC��@Ɛ\BCFA�p�            #      �         e   	      "   F      �   (               ;   $                  (      +                  $                  4   =                  
   S      (   B                        %         3         )   A      C      !            !                  #   -      '               #   !                     %         #            +                                                      '            !                                 -      #                  !                              !                           NmP�N�`�NHN�o�N/��O xN�Y�O<tO���NO�N���O��P�O3cO�*yOOmO�jeN�*�N_-SOx�;OY��OT��N�lM�QOl�N,��O���P(B�N��O���O)SvO��OeE�Nك�O,�OЙ�N�^WNH�N�βN�N��O��|O��pO�2N7��O�N xtN�;{N{s�O��O	�BO(�.O��TN>�N���  �  j  i  M  F  U  �    	k            �    ~  S    T  	  �  �  �  �  :  �  J  �  �  �  �  r    �  P  �  f    �  �  �  /  �  �  �  .  �  G  	�  T  
�  
�  q  ���P���ͼe`B;�o%   =���<D��<t�=q��;��
<o<�j=0 �<e`B=Ƨ�<�`B<�o<���<�t�<��
=8Q�=#�
=#�
=+=\)=t�=�w=�w=#�
=8Q�=,1=,1=,1=,1=H�9=8Q�=@�=D��=H�9=P�`=T��=}�=��=e`B=m�h=}�=�o=�7L=�7L=Ƨ�=�t�=���=� �=���=� ����������������" #09<C=<20#""""""��������������������**./<HKRKH</********()6BLLFB:6/)((((((((�������

�����}~�����������������������������������������),57873)������������������������������������������}z|����������������}���)5BKOOMB5)������!	��� ����)DNSQGB5 NPY[chtz���~yth[WPON�����1@JJLHA5)��"/;HIHB;8/"YX[bgstzwtg[YYYYYYYY�������������������������������:88:=@EHUahnnkdaUH<:���������


��������������������������*6COWZZXOC6*)6BFB?6))���������������������������
!*>@9(#
��	

								������5@CA5)����feeehmz���������zmff�������������������������� �������������������������������
��������������������������������������������������������������TMNUbcnqtsnbbUTTTTTT���"���������������zuyz������������������

��������6BIORSOA6)�	
#$-,,*'#
htutqh[WPX[hhhhhhhhh]ht��������������tg]mmnvz|{zynmmmmmmmm��������������������)6:63,) xyz��������������~zx�~������������������������

�����}~�������������������

���������)15652)�Ľнݽ��ݽнĽ½��ĽĽĽĽĽĽĽĽĽĻx���������������������x�u�v�x�x�x�x�x�x������������������������������������������������������������������������ּ׼ؼڼּʼ��������ʼռּּּּּּּ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ܹ������
��������޹ݹܹ۹ܹܹܹ�����������������������ĸĳıĵĿ����������#�0�<�E�G�F�@�<�0�#��
����������
�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�Eٽ�����������ݽнҽݽ�������M�Z�f�s�����������s�f�Z�M�I�C�D�A�K�MƧƳ��������������������ƧƚƆ�y�o�uƎƧ�����������#�����������������������������)�:�A�B�8�������������������M�Z�f�j�s�v�u�s�i�f�Z�M�A�6�1�/�4�A�G�M���ݿ��������ݿѿĿ������������������T�a�j�m�p�x�y�p�m�m�a�W�T�N�I�J�T�T�T�T�y���������������y�v�r�w�y�y�y�y�y�y�y�y�;�G�T�`�m�x�z�m�T�G�;�.�"�����"�*�;�6�B�J�O�P�L�K�C�?�6�)�$������#�)�6�a�m�z���������������������z�m�i�`�\�^�a����������������������������x�t��������A�N�R�Z�Z�\�Z�Z�N�I�A�@�A�A�A�A�A�A�A�A�`�m�y�������������y�m�`�U�M�I�H�K�T�X�`�'�'�-�3�/�'������'�'�'�'�'�'�'�'�'�������������������������������v�j�l�s����"�;�`�y�������y�m�T�;�.���߾������������ʾ̾ʾ��������������������������
��#�/�4�3�,�&���
�����������������
ŠŭŹ����������������ŹŵŭŠŜŕŕŠŠ�Ŀѿݿ�����(�4�*�/�(�����Ŀ����������������������������������z������������Óàìõù��ùùìàÓÇ�~�~ÇÎÓÓÓÓ�5�;�B�I�B�5�)����������� �����)�5���������������������������v�q�s��������-�3�:�F�R�S�T�S�F�:�-�$�!��!�"�-�-�-�-��������s�f�d�b�f�s�|����������������������������������������f�p�j�j�i�f�`�Z�P�O�O�V�Z�b�f�f�f�f�f�f�
�	�������
������!��
�
�
�
�
����"�-�/�6�>�A�@�7�/�"��	������������лܻ�������������ܻлǻ������Żо(�4�A�M�Z�^�r�s���s�Z�M�A�4�,�(� �$�(��߿ݿؿѿοοѿݿ������������������������������������y�v�|���������������������������������������������������������������������|�y�u�y���������������������������������������������������������ɺ̺պӺϺͺɺź����������������������(�5�A�N�Z�[�Z�X�J�A�5�.�(�%�����"�(E�E�E�E�E�E�E�E�E�E�E�E�E�EuEtEiEmEuExE���'�@�M�Q�V�V�H�4�'��������������C�D�O�\�`�\�[�O�I�C�B�B�C�C�C�C�C�C�C�C�g�t�w�}�t�k�g�f�e�g�g�g�g ? $ ^ & a % 1 3 , [ U  / Z $ * . U ' ; Q 5 4 r 7 n : 7 j = ) X L 3 b Y > u A � � < A Y R d Z B 6 J A 8 0 Y `    �  �  P  �  u  U  �  �  )  �  �    [  �  �  P    M  s  �  �  �    L  �  �  �  �  \  N  g  �    �  �  ,  �  �  �  ;  ^    M  o  H  �  >  �  �  a  C  k  Q  B  �  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  D;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  H  $  C  K  S  Y  _  c  g  i  j  i  e  [  Q  F  5  !  	  �  �  �  i  X  H  8  '      �  �  �  �  �  �  �  �  �  y  n  b  V  �  	  N  �  �     &  A  H  M  J  ?  (  �  �  I  �     p  �  �        #  *  1  8  @  K  `  e  X  -  �  �  �  �  }  d  ;  �  �  s    _  �  �  5  U  D    �  7  �  �  7  {  
0  �  z  �  �  �  �  �  �  �  �  }  U  (    �  �  W       �   �  �  �  �        �  �  �  �  �  r  ;  �  �  o    �  �  �  A  �  @  �  �  �  �  	1  	T  	h  	g  	\  	  �  H  �  %  V    �    �  �  �  �  �  f  A  %  
  �  �  �  �  �  ]  7     �   �          
  	                           $  )  U  �  �  �  �  �  �    	  �  �  �  w  N  &    �  �  !  �  �    	        �        �  �  �  l  (  �  R  �  �  �          
  �  �  �  �  �  �  �  �  t  Z  =  �  �  I   �  	�  
�    .  L  /  �  l  �  �  �  �  �  k  �  �  
G  �     �  @  �  �  �  �  �  �    �  �  �  �  �  q  C  �  t  �  �  (  ~  {  q  e  V  C  (    �  �  �  �  �  �  {  Q    �  ?   �    .  C  R  P  F  :  -      �  �  �  _    �  I  �  �  ,                     �  �  �  �  �  �  �  �  �  �  �  T  O  E  ;  /  !    �  �  �  �  �  b  ;    �  �  �  Z  #    K  �  �  �  �  	  	  �  �      �  3  �  ^  �  �  f  �  [  �  �  �  �  �  �  �  �  �  f  0  �  �  �  9  �  �  '  n  &  R  u  �  �  �  �  �  �  �  �  ~  a  ?    �  <  �    :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
    *  �  �  {  s  i  \  M  ;  &    �  �  �  y  E    �  h   �   Q  :  #    �  �  �  �  �  n  L  )    �  �  z  M  !   �   �   �  �  �  �  �  �  �  �  �  �  �  �    l  T  5    �  �  $  �  J  8  "    �  �  �  d    �  m    �  w  o  m  7  �    :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ]  '  �  �  �  {  :  �  ~  �  +   �  �  �  �  t  a  N  ;  )       �  �  �  �  �  h  K  *  �  �  �  �  �  �  �  m  V  =  "    �  �  �  �  n  W  7    �  �  r  ]  I  3    �  �  �  �  Y  0  
  �  �  V    �  y  >  E    �  �  �  �  �  l  R  8      �  �  �  �  j  B    �  �  �  y  c  X  d  �  �  ~  o  T  )  �  �  e    �  �  h  %  �  P  7    �  �  �  �  �  �  �  P    �  �  J    �  �  ?  �  �  �  �  ~  s  r  r  �  �  �  �  �  �  �  z    �  }  2  �  f  M  4      �  �  �  �  �    i  T  M  b  v  �  �  �  �      �  �  �  �  �  �  y  g  U  B  0         �  �  �  �  �  �  �  �  �  �  �  �  s  W  7    �  �  �    [  :     �  �  �  �  �  �  �  s  e  a  \  ]  b  h  a  A  !    �  �  �  j  �  �  �  �  �  �  }  Y  +  �  �  Y  �  �    d  �  <  �  �  �  �  �    #  /  '  	  �  �  *  �  -  �    �  t    8  �  �  q  ^  J  6  $    �  �  �  �  a  (  �  �  I  �  h    �  �  �  �  �  �  �  �  �  p  `  P  ?  .      �  �  �  �  �  �  �  �  x  j  [  D  &    �  �  b  A  "  �  �  �  \  �  .  .  .  .  ,  %          �  �  �  �  �  �  �  �  �  �  s  �  �  �  �  �  �  �  �  �  �  ~  d  D    �  �  U  �  �  G  9  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	`  	�  	�  	�  	�  	�  	�  	�  	�  	E  �  �     T  �  �    �  T  J  =  /  !    �  �  �  �  }  N    �  �  �  [  -    �  
�  
�  
�  
�  
�  
g  
2  	�  	�  	?  �  [  �  ]  �  ?  �    �  F  
�  
�  
�  
�  
�  
p  
C  

  	�  	z  	*  �  q    �  �  4    �  �  q  h  `  X  M  9  %    �  �  �  �  �  �  �  �  k  O  4    �  �  �  }  ~  {  ]  =    �  �  �  �  h  A    �  �  �  �