CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��1&�y        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�
   max       P�X�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =���        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F�z�G�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @v�\(�     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @O�           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @���            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >'�        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B32�        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�K�   max       B3>�        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ? �}   max       C��        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?փ   max       C��N        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          q        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�
   max       Px��        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?�1&�x��        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       =�F        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @F�z�G�     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?θQ�     max       @v��\(��     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @O            �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @��             U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?t   max         ?t        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��+I�   max       ?�,<�쿲        W�                        ,   +   *      '      $   g            9             ?   	               
   2         	               $      ,         
            !   
      
   !                  	   p               9   ;Nw��O�ӒN$mN&9�N0��N�O���P�[O��PH@ZN�B O��RN��OV,�P�X�O!2N&:Nn�P2ԣO���NNJbO)��PJΝN�D|P�N���N�N�>�O
i�O�%�N_tM�
N�~wN��9NF��NX�Op�Px��O��P'X�O,��O��N��2N�mDN�xN�O�m(N��lO8�Ns�O?��Oc �O���O='WN�S�O>��N��O�g�N�~�N�YO+�nO;+O�� O�\ͼ�`B��1���
����e`B�49X�D���D���D���o%   %   :�o:�o;��
;ě�<o<o<#�
<#�
<#�
<49X<T��<T��<e`B<e`B<u<u<��
<��
<��
<�9X<�9X<�9X<�j<ě�<���<�/<�/<�/<�h=o=o=o=+=C�=C�=C�=C�=\)=�P=�w=,1=<j=D��=D��=T��=T��=aG�=�o=�\)=�j=ȴ9=�����������������������;11;<@Ibnr|��zqnb_J;�������������������� #0<<E<0#          �|�����������������������������������������������������������������
�������
%/3<CEEC</#����)/;HR\bb]`TH<"	�(#)/5BN[][ZNKB65)((ggq�������������tnhg��
#')$#
 �����E@AHUanz������zsneSE)B[��������gSB)������������������������������������������������������������tt���������������|t������
�������$)0)+'((-/<HLU[_^UQH</++�� 6BMNZYPB)����<::;>ABNSWVVVTNB<<<<�� )5BIT[`b`B5)��)1.+)'tt������������}xvttt��������������������[YYWYamz�����}zmia[[���������������������������������������������������������

#/270/#






�����������������������������������������������������%$!).5BBELNRMBA51)%)N[t��������{[B}}~����������������}B=<?HUan���������aQBb`dhmu�����������uhb�������!#����369966<BKO[aa\OGB963rqtw����������wtrrrr#()-.367864)bgmt����tgbbbbbbbbbbfkp������������~togf][`anz~��zsnja]]]]]]���������� ����).66976)'����������������������#05;:43/%#
�����������	
������{}���������������� #%/166/*#      ��������������������
$),-)%$������

������" !"#%08<IJLIDD<50#"��������������������C@?AGHRT\aeghgdaZTHC+.269@BJO[bfe[XOB6)+����)6BOhrtdB6��������
%'
������������������������ĿķĿ�����������������ػ����������ûͻӻû��������x�l�j�i�l�x���F�S�_�l�l�l�c�_�S�I�F�A�F�F�F�F�F�F�F�F�нݽ��ݽܽнƽͽϽнннннннннй������������������������������������������������������������������������ŭŹ����������ŹŭŠŔŅŁŀŃŋŔŠŦŭ��(�5�N�Y�_�_�Y�N�A�(����ݿɿɿֿݿ�������������¿����������y�G�;�/�2�;�T�m���H�a�m�z�������m�a�T�;�"�������������#�H�)�6�B�I�O�T�R�P�P�O�F�B�9�6�2�)�)�)�)�)�f�s�������������a�M�A�6�4�8�:�F�M�Z�f��������� � ����������������������������F=FJFVF^FeFcFYFJF=F$FFFFFFFF$F1F=����"�.� � ���������s�g�N�E�F�Z�`�p���������	��"�&�/�5�/�$�"��	��������������m�y�������|�y�m�`�]�`�d�m�m�m�m�m�m�m�mE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Z�s���������������g�Z�A�5�(�$�*�,�4�A�Z��(�4�A�N�S�Y�V�N�A�(�������������Z�f�j�i�f�a�Z�O�N�S�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�������������������������������޾���	�G�Z�Y�1�"����׾ʾ����������������Ŀѿݿ����������ݿѿĿ����ĿĿĿ��h�uƁƘƦƫƷƿƳƧƜƎ�u�h�R�L�H�M�\�h�������������������������y�x�y�}�����������������½�����������������������������������'�'�)�'����������������������������������������������������ƶ�������������%�#���������øìâìù�������������������������������������������ҿ��ĿſǿĿ������������������������������T�a�l�m�q�o�t�m�a�T�T�P�P�T�T�T�T�T�T�Tìù��������������ùìàØßàçìììì����!�'�"�!��������������ù����������ùïìàìóùùùùùùùù���$�0�9�=�E�C�=�0�$��������������tª¬±¥�g�N������W�e�c�t���)�.�5�2�)�#������������������(�5�A�N�f���������s�r�d�Z�N�0�������(�ʾ׾�������������׾Ӿʾ������������������������������������������x�u������лԻܻ�����	���������ܻջл˻м�'�+�4�@�B�F�H�@�4�*�'� ��������������������ûƻû��������������x�|�x���;�A�G�R�L�G�;�8�5�7�;�;�;�;�;�;�;�;�;�;�����������¿¿������������|�o�m�c�m�w���4�A�M�O�Y�U�M�A�4�)�(�'�(�+�4�4�4�4�4�4�(�4�M�Z�f�n�s��}�s�Z�M�A�4�(�����(����������������������������������������ÇÓÛÜÓÊËËÇ�z�n�a�U�S�Z�a�j�nÄÇ�����*�+�$�!�������ּԼռڼ��������	��&�+�'�"��	����ʾ��������;۾������(�4�7�A�D�A�6�4�(����������Z�f�s�����������s�f�\�Z�X�Z�Z�Z�Z�Z�Z�ܹ������'�+�'�������ܹϹùƹϹ��
��#�0�;�<�I�L�I�F�<�0�)�#���
�
�
�
D�D�D�D�D�D�D�D�D�D�D�D�D{DsDoDtD{D�D�D������ɺֺ޺�������ݺֺɺ����������y�����������������z�y�l�i�b�l�u�y�y�y�y���
��#�0�1�<�A�I�R�I�0�#��
�������������������ɺֺݺֺҺɺ�������������������������������ʼͼʼ������������y�w�y�y�E�E�E�E�E�E�E�E�E�E�EuEqEpEiE\EHEPEiEuE� Z * 9 p 5 | , E O < e + 1 l A S d * , ? X ! X b 9 9 $ ; 4 < L 0 T 8 L C 0 s %   1 ! m _ D Z @  i H ~ " U 5 < O V  M C d ?  U  �  [  H  Z  H      �  &  �  #  �  �  -  �  J  J  u    ]  |  e    &  ^  �  	  �  <  �  �    �  �  {  g  Q  9  M  �  o    =  �  /  [  �  �  �  ~  &  �  �  �  �  �     )    �  �  �  �  ��ě��ě���o��o�t���`B<49X=#�
=#�
=�w<#�
=�w<o=�P=�"�<���<49X<�9X=�7L<��<�o=,1=���<�9X=��<�1<�1<�j<�h=�O�<�<���<�=D��<��=\)='�=}�=<j=�\)=#�
=P�`=,1=0 �=<j=��=�7L=0 �=Y�=8Q�=�O�=q��=ix�=m�h=e`B=���=y�#>'�=�%=���=\=�F>�->%�TBc�B'i�B#P�B%�4BF Bw�B-�B��B�$A���BZ%B_6B:Ba`B��BDSB�zB��BMrB�FB]B�RBۏB B�B��B)rB!~A�w�Bs�B:�B�Bb�B!ߨB��B��Bd�B�B�B�B32�B�B�%B�B��B	��BJB�	B#O�B>�BI�B$�pB#4B}B�,B��B'�B�B%�tB,^�A���B^�B�BFB��B'A�B#C�B%}	B=�B@xB4!B��BŮA�K�B��BB�B?dBA�B?�B?�B~�B��B �fB��B?BŻB �B-OB��B�yB?�B!�RA�nB@B�CBL�B��B!��B��B� BBKB
Q8B
�-B��B3>�BBG�B8�B'&B	�\B@�B�B#DzBC�B@	B$��B#:&B<�B�B~�B?�B?�B&8�B,6A��B>@B?B��A�c@���@�.�A)��?@�	A��A��A���Ak�+A�p�A�YABUA�uzC��A�a�A�D�AlQ�C�j�A�bA��zA?5�A�l*AS�A}�qB�sAqv�A!!@��B�?A�zJA�%Aw5SA�Z�A�"Y@bW�A͹�B	^"A�_A��A��&AS��A���@���@�F@�k�Ac�Aqr�A:��A;��A��A��An�AV�A4G�ABپ? �}A���C���@7�"A�A�WM@"v@�EC��A�h@�E�@��A*�?N�A���A��A���Al�A�p:A��AB��A���C��NA�sPA���Ak�fC�h A���A�W�A?SA��AR��A�CB��Ap�A$�@�B-CA�z�A�AwOA�~Àk@cޘA͉=B	��A�ٚAՀ�A��dAS��A���@��h@���@�̞Ac�VAs%�A;U3A;�WA��eAȁkA� AW�bA2�AB�4?փA낗C��@CCrA%A�|{@ O�@���C�                        ,   ,   *      (      %   g            :             ?   
            	   
   2         	      	         $      ,                     "   
         !            	       
   q   	            9   ;                        '   #   +      !         A            +            5      %               !                        9      )                                       !                              %   #                              %                                                                                    9      )                                       !                              #   Nw��Ol�N$mN&9�N0��N�Ob��O���O#C~P'�N�ӍOD�N��O�zO���N�/N&:NT�EO��OG��NNJbN���O���N�D|O���N���N�N�>�O
i�OsP�N_tM�
N=D�Ne@nNF��N#ƅN�:Px��N�F P;O,��Oh5N���N2zN�xN�O�KJN��lO8�Ns�N0�@O<.�O���O='WN�S�O�2N��O�UN�~�N�YN��O;+O���OFE�  �  N  k  G  {  .  �  �  �  �    �  b     c  �  �  {  �  }  �     �    �    �      j  s  �  �  �  z  r  �  B  h  	  �  d    �  c  �  R  "      
  �  �  �  �  �  �  5  t  i  .  B  �  M��`B��t����
����e`B�49X��o;��
<�o<#�
:�o<u:�o<o=}�<#�
<o<t�<�/<u<#�
<�9X=�w<T��<�j<e`B<u<u<��
=o<��
<�9X<ě�<�<�j<���=o<�/=o<�h<�h=C�=+=\)=+=C�=t�=C�=C�=\)=e`B='�=,1=<j=D��=aG�=T��=�^5=aG�=�o=��P=�j=�"�=�F��������������������54?IUbenrxzytnbQIE<5�������������������� #0<<E<0#          �|����������������������������������������������������������������	������	
#/6<<<<;6/#	
/;?MTXVWVTH;"$)05BLNVNHB5)s}���������������yts��
#')$#
 �����EHUaknvz}���zynjXUJE1-,.5BN[w���|tg[NB71������������������������������������������������������������{{|����������������������
����$)0)./0<HNUVUQH<<<0/.... )39BFD>61<::;>ABNSWVVVTNB<<<<	)15BNQUTNB5)')1.+)'tt������������}xvttt��������������������[YYWYamz�����}zmia[[������������������������������������������������������������
#/0/'#
�����������������������������������������������������)')5BGJEB5*))))))))))N[t��������{[B��������������������C?>@HUansz�������aRCb`dhmu�����������uhb������� ����66=BLO[__[[ONGB;:666yz����������yyyyyyyy#()-.367864)bgmt����tgbbbbbbbbbbgklr�������������thg][`anz~��zsnja]]]]]]���������� ����).66976)'����������������������
#06621-#
 ����������	
������{}���������������� #%/166/*#      ��������������������
$),-)%$������

�������" !"#%08<IJLIDD<50#"��������������������EBACHKTaceeba^WTOHEE+.269@BJO[bfe[XOB6)+� )6B[hnnh[B6���������
  
����������������������ĿķĿ�����������������ػ������»Ȼǻû����������x�q�o�x���������F�S�_�l�l�l�c�_�S�I�F�A�F�F�F�F�F�F�F�F�нݽ��ݽܽнƽͽϽнннннннннй������������������������������������������������������������������������ŭŹ����������ŹŭŠŔŇŅłŅŇŏŔŠŭ�(�5�A�J�S�R�L�A�5�(�����ݿտ׿޿���(�m�y���������������z�y�m�f�`�T�L�N�Z�`�m�H�T�a�h�r�p�m�T�K�;�"�������������/�H�6�B�H�O�S�P�O�N�B�<�6�3�*�+�6�6�6�6�6�6�f���������������s�f�Z�X�N�J�M�P�Z�c�f��������� � ����������������������������FJFVF`F^FVFTFJFBF=F1F$FFFF!F$F1F4F=FJ�������������������������������������������	���"�#�"���	���������������������m�y�������|�y�m�`�]�`�d�m�m�m�m�m�m�m�mE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��N�Z�g�s�����������������s�g�Z�K�<�8�@�N���(�5�A�F�N�R�N�J�A�5�(���������Z�f�j�i�f�a�Z�O�N�S�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z������
����������������������������������ʾ׾�����	����׾ʾ��������������Ŀѿݿ����������ݿѿĿ����ĿĿĿ��\�h�uƁƎƕƚƞƛƒƎ�u�h�d�Y�R�O�O�W�\�������������������������y�x�y�}�����������������½�����������������������������������'�'�)�'����������������������������������������������������ƶ��������������������������þõù���������������������������������������������ҿ��ĿſǿĿ������������������������������a�f�m�n�m�k�b�a�`�T�S�R�T�Z�a�a�a�a�a�aùý��������ùìàÞàçìñùùùùùù����!�'�"�!��������������ù����������ùùìäìùùùùùùùùù��$�*�.�-�$���������������tª¬±¥�g�N������W�e�c�t����)�1�.�)����������������(�5�A�N�c�~�������~�s�n�Z�N�3�	��� ��(�ʾ׾�������������׾Ӿʾ������������������������������������������z�x������������������������ֻܻܻ�������'�4�>�@�C�@�@�4�0�'�'�#�'�'�'�'�'�'�'�'�������������ûƻû��������������x�|�x���;�A�G�R�L�G�;�8�5�7�;�;�;�;�;�;�;�;�;�;�����������������������������}�q�p�j�y���4�A�M�O�Y�U�M�A�4�)�(�'�(�+�4�4�4�4�4�4�(�4�M�Z�f�n�s��}�s�Z�M�A�4�(�����(�����������������������������������������zÇÉÇÄ�~�z�n�a�a�a�n�n�x�z�z�z�z�z�z������!�&�%��������ּ׼ܼ���������	��&�+�'�"��	����ʾ��������;۾������(�4�7�A�D�A�6�4�(����������Z�f�s�����������s�f�\�Z�X�Z�Z�Z�Z�Z�Z����������������ܹϹȹ˹Ϲܹ���
��#�0�;�<�I�L�I�F�<�0�)�#���
�
�
�
D�D�D�D�D�D�D�D�D�D�D�D�D�D}D�D�D�D�D�D������ɺֺ޺�������ݺֺɺ����������y�����������������z�y�l�i�b�l�u�y�y�y�y���
��#�,�0�:�:�0�#��
������������������������ɺֺݺֺҺɺ������������������������������Ƽʼ����������������}�|�����E�E�E�E�E�E�E�E�E�E�E�ExEuEuEuEsEnEuEvE� Z * 9 p 5 | & B ' F S  1 c 5 F d ' 0 ? X # 0 b 0 9 $ ; 4 A L 0 Y 7 L G % s  " 1 " Y V D Z ?  i H k  U 5 < > V  M C P ? v R  �  �  H  Z  H    �  �  _  j  �  U  �  u  �  �  J  a  �  �  |  �  e  &     �  	  �  <  �  �    �  w  {  ;  �  9  �  �  o  �  �  n  /  [  �  �  �  ~  w  �  �  �  �        3    �  2  �    �  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  ?t  �  �  �  �  �  �  �  �  �  �  �  u  i  \  P  ;  "  	   �   �  7  @  G  L  M  M  I  C  ;  1  %      �  �  �  �  [  1    k  d  \  U  M  C  1      �  �  �  �  �    d  G  *     �  G  C  @  <  9  5  2  ,  &            �  �  �  �  �  �  {  m  _  Q  B  1  !      �  �  �  �  �  �  �  �  �  �  �  .  (  #          �  �  �  �  �  �  �  }  j  V  B  -    �  �  �  �  �  v  i  Z  G  .    �  �  �  �    Y  6  )  E    g  �  �  �  �  z  X  3    �  �  y  D    �  N  �    <  f  �  �  �    *  G  j  �  |  Z  +  �  �  T  �  o  �  �  n  �    U  �  �  �  �  �  l  %  �  w    �  �  �  H  �  �  @          �  �  �  �  �  x  S  +  �  �  �  ;    �  �  �  N    �  �  �  �  �  �  �  �  �  �  �  h  )  �  �  ?  �  �  b  [  T  M  F  ?  6  -  #      �  �  �  �  �  �  f  <    �  �  �  �  �  �  �  �  r  ^  ,  �  �  ,  �  v    �    u  I  �  H  �  �  �  �      ;  Z  a  ?    �  g  �  d    @  �  �  �  �  �  �  �  �  �  �  �  �  S    �  �  f    �  E  �  �  �  �  �  �    y  r  l  e  _  Y  P  D  7  *        z  {  w  s  z  s  h  Z  H  6      �  �  �  v  Q  -  	  �  #  `  �  �  �  �  �  �  �  �  �  w  O  7    �    _  �  ~  :  U  l  x  }  z  q  a  O  B  A  >  7    �  �  i    �  h  �  �  �  �  �  �  �  �  o  W  4    �  �  t  C     �   �   n  d  �  �  �  �          �  �  �  �  Q    �  l  *  
  �  �  4  k  �  �  �  �  �  �  �  �  �  �  h  1  �  d  �    +      �  �  �  �  �  �  y  e  Q  <  %    �  �  �  �  �  �  k  q  q  r  x  �  �  �  �  }  e  @     �  �  U    �  �   �            �  �  �  �  �  �  �  �  �  �  u  a  M  8  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  l  ^  P  B    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r      
    �  �  �  �  �  �  �  �  �  l  U  =  $    �  �  �  �  :  Z  h  g  V  =  %    ,  5    �  ;  �     Z  �  �  s  r  p  i  a  W  M  D  9  -  "        �  �  �  �  �  	  �  �  �  y  o  e  \  P  D  7  +        �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  h  ^  \  Y  Y  X  U  R  O  K  G  <  U  j  x  �  �  �  }  j  N  ,    �  �  q  7  %  �  �    z  k  \  ?    �  �  �  �  �  �  �  �  �  �  �  |  g  P  :  a  i  o  q  q  n  h  b  X  M  >  +    �  �  �  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  ]  6  �  �  [  F  B    �  �  �  {  s  U  )  �  �  �  �  �  J    �  �  Q  �  R  \  c  h  h  h  e  `  ^  P  :      �  �  �  c  L  \  �    	    �  �  �  �  ]    �  �  H    �  v  1  �  �  �  e  �  �  �  �  �  �  �  n  V  =  #  
  �  �  �  �  �  d  %   �  U  b  d  _  Z  P  ?  -    �  �  �  �  o  B    �  P    �  �  �    �  �  �  �  �    ]  :    �  �  �  i    �  U   �  �  �  �  �  �  �  �  �  �  �  �  �  i  ?    �  �  �  �  L  c  L  ;  /  ,  +  +  !  
  �  �  �  u  I    �  �  t  X  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  G  Q  G  ,    �  �  �  o  X  3    �    9  �  �    l  ~  "      
    �  �  �  �  �  �  �  �  |  ]  >    �  �  �        �  �  �  �  �  |  ?  �  �  A  �  �  ^    �  �  =          �  �  �  �  �  �  �  }  g  P  #  �  �  u  Y  @  �  �    A  �  �  	�  	�  	�  	�  	�  
  	�  	  �  u  �    "  &  �  �  �  �  �  �  ~  r  g  ^  P  <  $  
  �  �  u  �  h   �  �  |  h  S  >  *    �  �  �  �  o  E    �  �  �  u  T  B  �  �  �  �  �  �  |  i  W  D  *    �  �  �  f  7    �  N  �  v  d  N  8       �  �  �  �  h  @    �  �  K        �  2  R  �  �  �  �  �  �  `  0  �  �  s  *  �  �  4  �  M  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  n  c  X  L  A  9  �  F  �  �    /  2    �  �    `  �  �     
  
�  �  �  t  `  L  4      �  �  �  �  |  d  K  -    �  �  �  �    i  Z  K  9  #  	  �  �  �  }  W  .    �  �  �  y  _  A  !  �  �  �  +      �  �  �  Z    �  �  +  �  h    �  �  �  B  (    �  �  �  �  �  �  r  O     �  �  b    �  L  �    �  �  �  �  �  P  
�  
�  
P  	�  	�  	  |  �  (  |  /  -  +  i  
�  
p  
�    8  K  &  
�  
�  
l  
#  	�  	�  	X  �        �  