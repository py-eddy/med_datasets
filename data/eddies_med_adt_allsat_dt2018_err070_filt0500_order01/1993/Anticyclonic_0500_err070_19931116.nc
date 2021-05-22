CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���
=p�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�HQ   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =ě�      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @E�=p��
     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ҏ\(��    max       @vo��Q�     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @�%@          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >D��      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�R�   max       B,,      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B,9W      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?$��   max       C�a�      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C�f�      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�HQ   max       PsMe      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?��Q�`      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       =�S�      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E��Q�     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ҏ\(��    max       @vo��Q�     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @��           �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?{qu�!�S   max       ?��
=p��     �  N�            
   +            O            \         	            ,         s          
      '               
            
   B         3   	            
   �                  9   N��N6��O��OK_XO�N��P#�'N.�P N'jBNIO_��Px�~N5�N^��N��M�HQObPQO��O)�MN6��OF�P���O�6O��O-)Nn#�O��LN9�O�TLO"UGN���N���O��O� �OP��N�[�P)%7NP,OFbZP �-O 9N�
�N�|]O*olNy5�O�a�O��N� 1Nw`~O$�eN��BO:��N`����`B���
:�o;�o<o<#�
<D��<e`B<e`B<�C�<�C�<�C�<�C�<�t�<�t�<�t�<�t�<�1<�9X<�9X<���<���<���<���<���<�/<�/<�`B<��=C�=C�=\)=t�=�P='�='�=0 �=0 �=49X=@�=ix�=m�h=q��=y�#=}�=�%=�%=�7L=�C�=�O�=��=���=� �=ě���	"#)&"		�������������������������	5>?CNgjj[N8)����
#'%&)'!
 �����
+.-)#
�����().6<A=6651)((((((((qq����������������{qegt������tjgeeeeeeee����������������#),+)(������������������������������������������ O[juxqhO6"������������������������������������������,/4<FHPUY^[URH<70/,,HFFFHTNUVUJHHHHHHHHH��������������������,/2AHU\aekmkeaUH<6/,������������������������� �������������#/<HRUXUJH</)$#��)Bg��������a[��ehx��������������tmeYZ[\`hmtx�{thgc][Y<75<=HHLUaabijfaUIH<lgnoz�����znllllllll��������������������/./:<<HRQH</////////@9?IN[gt����~utrg[K@�����������������������������������������������"#0<>IKIIA<70#y�����������������y��������������������RZ[gqtv����~ztig\[RR����)6O\^aTNB)��������������������������)+5:>5)��)�����6BJKDA6)
#/<HUa]H/)#
����������������������������������������")69BOYe[UOB=6)%345BNPQONB9533333333jhks��������������tj)558::;:5)!��������������������	
#$,/3/#
				�{}��������������������������������������������bcknz~�~znbbbbbbbbbb�a�c�h�m�r�n�m�a�T�P�H�C�H�H�T�^�a�a�a�a�m�z���{�z�m�g�a�X�Z�a�h�m�m�m�m�m�m�m�m�`�m�y�������ÿ������������y�r�\�T�S�W�`�_�e�l�x���������x�l�_�S�F�:���-�:�F�_�[�g£�g�N�B�5�)�!����)�/�B�[ÇÌÓÖÓÇ�z�n�k�n�zÅÇÇÇÇÇÇÇÇ������������������������h�s�v�}��������������������������������������������������(�4�A�R�U�S�L�4�(������߽��� �����������������z�y�z�~���������������������������������������������������������˾Z�f�s�s�w�����������s�f�Z�M�>�8�>�M�Z�����׾��������׾��f�M�B�A�Q�f���������C�O�\�h�m�i�h�\�O�M�C�A�C�C�C�C�C�C�C�C�T�a�b�h�m�n�m�f�a�_�T�P�M�R�T�T�T�T�T�T�������������������������������������������)�5�8�5�)�$���������������ĿͿѿҿ̿ƿ��������������������������	���"�&�*�/�1�/�"��	��������������	ÇÓàìôùþÿùìàÔÓÒÇÁ�|�}ÃÇƎƚƧƨƭƧƚƎƌƋƎƎƎƎƎƎƎƎƎƎ���$�%�%�#�$����������������������A�Z�������������k�N�5���������(�A�s�������������������f�Z�^�^�a�h�d�f�s���������ʼмʼ�����������y�~������������������������������������������~�������a�m�y�z��z�y�m�a�^�]�_�a�a�a�a�a�a�a�a����4�E�F�M�Y�e�Y�C�4�'�������²¿����������¿²±«²²²²²²²²²���$�0�5�8�3�&����������������������4�@�B�D�G�I�B�@�4�'�#�������'�,�4���!�-�:�?�:�:�-�!�����������G�T�`�f�l�`�T�G�;�6�;�?�G�G�G�G�G�G�G�G����������������ּҼʼ��Ƽʼּ׼��O�f�e�f�d�Z�O�6�)�������������)�B�O�U�a�h�i�e�a�]�U�L�H�<�8�/�!�#�<�>�H�N�UƎƚƛƣƧƨƧƜƚƏƎƁ�u�o�u�vƁƄƎƎ�!�-�F�S�d�i�r�v�m�_�S�:��������� ��!E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;M�Z�f�i�p�s�t�s�t�s�n�f�Z�O�C�>�A�F�B�M�@�E�L�Y�b�r�������ź˺ĺĺź����~�r�S�@�����"������������������������������������ּӼʼʼּ��ｅ���������������������y�t�m�p�q�y�|�����m�z�����������������x�m�f�b�a�^�_�a�h�m�{ŇŊœŉŇ�{�n�k�b�n�t�{�{�{�{�{�{�{�{�)�6�B�O�\�d�g�g�[�O�B�6�/��������)�N�[�g�t�u�t�g�[�N�B�5�,�-�5�B�D�N�N���������������ܹϹ͹Ϲܹܹ���`�m�o�o�m�b�`�T�M�G�=�A�G�Q�T�_�`�`�`�`�'�4�@�E�G�H�C�@�4�0�'����
����%�'��������������������������������������EiEuE�E�E�E�E�E�E�E�E�E�E�E�EzEoEiEeE_Ei���������������������������������������� N \ : b N T 6 \ . Q m ` < P T 0 r i ?  F 4 4 o P * = > N 9 5 Z . * I Q 0 ' % N 2 � : h L 8  e H w F 9 7 /  �  W  �  �  �  8  �  �  �  R  Z    /  j  �  �  <  6  q  r  a  �  �    >  O    l  e  k  i  �  �  #  �  �  �  �  [  �  I  �    B  y  r  �  K  �  �  z  �  �  w:�o��o<�j<e`B=P�`<�t�=�w<�C�=�j<�1<���=49X=�/<�j<�j<�/<��
=�w=+=�+<�/=Y�>bN=m�h=0 �=t�=o=�7L=49X=]/=T��='�=<j=]/=�\)=}�=T��=�;d=ix�=}�=�"�=�+=���=���=�-=�t�>D��=��
=��T=��P=��=�Q�>n�=�A�R�B��B*rB$�VBmB��B��B	�LB"|DBDBc�B�cBaBUBjZBZBS�B��Bp�B"&@B��B��B~BthB��B�WBJRB!H�B�`B	(5B�HB\�BHB%��Bt@BdB	rjB)5B��B�bB�#B>xB]'B,,B�.B��B	B�DB��B{B��B_CB��B�A�~�B��B�BB$�jB?�B��BR\B
/~B"��B?�B��B�gB@rB@�Bx�B8=B�BB��B[NB"@(B��B��B�wB�-B��Ba�BG
B!D&B��B	)�B�bBI�B@�B%W�B�B�-B	��B��B�B7ZBֻB��B��B,9WB6B��B8�B�wBARB@�B<�BE�B�B�WA��A�^2An�@�Y!A��A��A��fA�lXA4ӰA��iA��IA?��AKH�B�A�$A��KA��At�jA��A�olB:TAӰA��HAD�2@��*A���A�\�@��A��BɄ@˧�@k��Af�iA��A�F�A�k~B��@w�/C�a�A>�@G~A2��A��AxA���A�A׈xA�?$��Ag�@��rA��|C��@�*iA���A�n�Ak_@�t^A���A��A��A�R�A4�*A���A�tsA>��AL�BOuA��A�zwA��CAt�A�|�Aˉ4BU�A�S�A��CAD��@�R:A��,A��@���A�y�Bͱ@��*@`�VAf�BA�IA�kA�}Bgg@| �C�f�A?�@��A1
mA��A<A���A�|AׁUA���?�Ai	�@��VA�~�C�@���   	            ,   	         O            \         
            -         t   !      
      (                           
   C         3   	            
   �                  :            #      #      )      '            7                              A   )                                 %         )         '                                                #            %                  !                              /                                             #         #                                       No�N6��O��OK_XOX�VN��O�*�N.�O�mN'jBNIO�O�jrN5�N^��N�0�M�HQN�:BN�YN�F�N6��O�PsMeO���N��GO-)Nn#�OuN9�O�p�O"UGN���N���N���O���OP��N�[�P�NP,OFbZO��<O 9N�
�Nw�O Ny5�OK5�N���N� 1Nw`~O$�eN��BOLDN`��  �  �  P  �  �  �  �  �  9  �  �    �  V  :  �  �  a  (  �  u  j  �  @  K  �  9  �  �  �  >  �  j  �  b  A  @  �  �  �  �  o  e  �  �  a     $  �  �  �  �  �  ����
���
:�o;�o<�1<#�
<�C�<e`B='�<�C�<�C�<�9X=T��<�t�<�t�<��
<�t�<�/<ě�=�P<���<��=e`B=o<�/<�/<�/=C�<��=\)=C�=\)=t�='�=H�9='�=0 �=]/=49X=@�=u=m�h=q��=��=��=�%=�S�=�C�=�C�=�O�=��=���=ě�=ě��	"%""	��������������������	5>?CNgjj[N8)����
#'%&)'!
 �����������

���().6<A=6651)((((((((zw����������������zzegt������tjgeeeeeeee���������� ������#),+)(����������������������������������������&%!)6BOY^fgd[OB6,&����������������������������������������./;<=HHKUXXUOH<:3/..HFFFHTNUVUJHHHHHHHHH��������������������//5<EHUahjhaaUH<;///������������������������� ������������� #/<DHORQHE</.(#(5Bg�������[B5}������������������}Y[]bhltv~�~|ytihc^[Y<75<=HHLUaabijfaUIH<lgnoz�����znllllllll��������������������/./:<<HRQH</////////B;@KN[gt�����ztga[LB�����������������������������������������������#/01;<B<:10)#����������������������������������������RZ[gqtv����~ztig\[RR���)BRWZWPIB6��������������������������)+5:>5)�����6BHIC@6)��
#/<HUa]H/)#
����������������������������������������$)6@BNO[ZSOB760)345BNPQONB9533333333}vtw���������������})3589::95)"��������������������	
#$,/3/#
				�{}������������������������������������� �������bcknz~�~znbbbbbbbbbb�T�a�e�m�o�m�f�a�W�T�H�F�H�L�T�T�T�T�T�T�m�z���{�z�m�g�a�X�Z�a�h�m�m�m�m�m�m�m�m�`�m�y�������ÿ������������y�r�\�T�S�W�`�_�e�l�x���������x�l�_�S�F�:���-�:�F�_�B�N�[�g�t�~�g�[�N�B�6�1�5�>�BÇÌÓÖÓÇ�z�n�k�n�zÅÇÇÇÇÇÇÇÇ���������������������������}�~������������������������������������������������������(�4�@�E�F�B�4�(��������������������������z�y�z�~���������������������������������������������������������˾M�Z�f�m�o�s�t�|�~�s�f�Z�W�M�K�B�<�A�C�M�������ʾ׾����׾ʾ����s�b�_�d�o����C�O�\�h�m�i�h�\�O�M�C�A�C�C�C�C�C�C�C�C�T�a�b�h�m�n�m�f�a�_�T�P�M�R�T�T�T�T�T�T�������������������������������������������)�5�8�5�)�$�������������������ÿ��������������������������������	����"�$�'�.�"��	��������������	�	ÓàìõùúùìèàÓÍÇÅÇÉÓÓÓÓƎƚƧƨƭƧƚƎƌƋƎƎƎƎƎƎƎƎƎƎ����� �������������������� ���(�A�Z�������������s�Z�N�7�0������(�s�������������������s�h�d�b�c�c�e�f�s�������ʼμʼ�������������z��������������������������������������������~�������a�m�y�z��z�y�m�a�^�]�_�a�a�a�a�a�a�a�a����1�4�@�H�M�@�6�'������������²¿����������¿²±«²²²²²²²²²���$�0�3�7�2�%���������������������4�@�B�D�G�I�B�@�4�'�#�������'�,�4���!�-�:�?�:�:�-�!�����������G�T�`�f�l�`�T�G�;�6�;�?�G�G�G�G�G�G�G�G�ּ�������������ۼּʼǼʼ˼ּּּ��O�V�[�]�\�\�O�D�6�)�!�������)�B�O�U�a�h�i�e�a�]�U�L�H�<�8�/�!�#�<�>�H�N�UƎƚƛƣƧƨƧƜƚƏƎƁ�u�o�u�vƁƄƎƎ�-�F�S�^�b�e�j�j�_�S�:�-�!�������
��-E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;M�Z�f�i�p�s�t�s�t�s�n�f�Z�O�C�>�A�F�B�M�����ºȺźú��º����~�r�X�E�M�Y�e�r���������"������������������������������������ּӼʼʼּ���y���������������y�w�q�s�y�y�y�y�y�y�y�y�o�z�����������������z�m�i�d�a�`�a�b�m�o�{ŇŊœŉŇ�{�n�k�b�n�t�{�{�{�{�{�{�{�{��)�6�B�M�O�V�Y�V�O�F�B�6�)�������N�[�g�t�y�}�t�r�g�[�N�B�5�-�.�5�B�F�N�N���������������ܹϹ͹Ϲܹܹ���`�m�o�o�m�b�`�T�M�G�=�A�G�Q�T�_�`�`�`�`�'�4�@�E�G�H�C�@�4�0�'����
����%�'��������������������������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E}EuEsEjEnEuEu���������������������������������������� + \ : b G T 4 \  Q m K B P T F r T L  F 0 @ ] M * = 0 N : 5 Z .  5 Q 0 ( % N 0 � : b G 8  b H w F 9 1 /  r  W  �  �  �  8    �  /  R  Z  h    j  �  �  <  
  6  �  a  /    �  .  O    �  e  /  i  �  �  �  .  �  �  A  [  �    �    �  6  r  �  1  �  �  z  �  9  w  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  �  �  �  �  |  ^  $  �  �  �  ^  1    �  �  �  �  �  �  �  �  �  }  o  a  S  E  7  ;  E  P  Z  d  P  8  #    �  �  �  �  �  l  v  |  i  R  '  �  �  �  �    �  �  �  �  �  �  u  a  O  ?  3  -  !    �  �  �  |  l  `    K  �  �  �  �  �  �  �  �  g  8    �  �  B  �  >  U  �  �  �  �  �  x  `  ?    �  �  �  �  i  A     �   �   �   o   C  �  �  �  �  �  �  �  �  v  ^  I  1    	  �  �  �  F  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    1  N    e  �  �    1  8  7  +    �  �  n  (  �  L  �  �    ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  R    �  �  G  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  ]  %  �  �  c    �    �   �  `    �  �  Q  �  �  �  �  �  �  w  <  �  �     �  �  L   �  V  J  >  2  %      �  �  �  �  �  �  �  �  p  i  e  a  \  :  8  6  5  2  -  '  "              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  h  X  `  h  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  0  6  0  8  I  Z  `  ^  T  ?  $    �  �  �  �  �  ^  4  x    !  %  '  $      �  �  �  �  {  S  "  �  �  �  T     �  i  �  �  �  �  �  �  �  �  �  u  !  �  .  �  �  D    �   �  u  t  t  s  r  r  q  p  o  o  j  `  V  M  C  9  0  &        A  U  a  i  h  ]  C    �  �  n  (  �  s    �  �  >  j  �  ?  �  �  �  �  �  �  ~  6  	  �  �  �  �    f  �  \  9  �    (  ;  =  .    �  �  �  �  d  -  �  �  ;  �  i  �   �  D  J  J  B  2      �  �  �  �  g  ?    �  �  d    �  C  �  �  �  �  �  �  �  �  �  �  x  h  X  D  '            9  7  6  4  1  )  !      �  �  �  �  �  �  �  k  M  .    �  �  �  �  �  �  �  �  b  O  G  >  1    �  �  q  �  m   �  �    r  k  [  J  8  &    �  �  �  �  �  w  T  .    �  �  �  �  �  �  �  �  �  �  �  �  v  a  G  #  �  �  �  p  C  4  >  8  5  :  5  .  &      �  �  �  �  �  ]  -  �  �  M  �  �    k  V  E  4  #        �  �  �  �  �  �  �  �  �  w  j  X  D     �  �  �  �  �  �  t  Z  =       �  �  �  b  .  {  �  �  �  �  �  �  �  �  �  |  q  e  V  A    �  �  Q  �    .  /  R  ^  b  W  2  	  �  �    B  .    �  �  j    �  A  :  .  !    �  �  �  �  �  T     �  �  h  !  �  �  �  �  @  9  2  )           �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  c  &  �  �  3  �  
  H  2  �  �  �  �  �  �  e  E  .    �  �  �  �  �  X  #  �  a  �  S  �  �  �  �  �  �  }  m  [  G  1    �  �  �  �  �  ,  �  _  �  �  �  �  �  �  �  ~  d  @    �  �  8  �  d  �  �  >  d  o  X  A  +    �  �  �  �  �  �  n  W  2    �  �  �  �  �  e  L  5  '    �  �  �  �  ]  .  �  �  �  l  >    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  F    `  y  �  �  �    r  f  X  G  0    �  �  p    �  '  �  *  a  S  E  4  #    �  �  �  �  v  L    �  �  �  b  /  �  �  z  K  �  6  �  �  �  �  �  �  �  &  �  �    �  �  	�  K  )  !  #  "        �  �  �  �  {  U  ,  �  �  �  9  �  �  O  �  �  n  X  B  ,       �  �  �  �  �  }  X  .  �  �  �  \  �  �  w  d  S  I  >  4  '      �  �  �  �  n  O  0    �  �  �  �  �    B     �  {  G  k  f  ?    �  �  j    �  M  �  �  �  �  �  o  X  @  #    �  �  �  L  	  �  [  �  �  %  �  �  �  �  �  �  �  X    �  9  
�  
)  	�  �  �    �  ^  �  �  M  (     �  �  r  J    �  �  S  �  �    �  "  �  $  �