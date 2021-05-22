CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��vȴ9X     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�k{   max       P��)     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =�w     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @Fe�Q�     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vj�G�{     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @R�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @���         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       =C�     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�H�   max       B0��     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0E>     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�q7   max       C�,c     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��   max       C�(O     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          >     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�k{   max       PE��     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?����C�]     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       =�w     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @FNz�G�     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vj�\(��     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @R�           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�&�         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =   max         =     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}}�H˓   max       ?����C�]     �  _�            	                                 	         
   
            %   >   "   	   2                  !      	                  
   &      %         "   9                  9   -         
               *         +               &      Nn��NM(BN�ArNm�:N�N~N�5�N��zN"��O.��O���N�MN��N�h�O7�zNJU�O��LO��xN���O+�	O�!�O��Nf��O��4O��O�CNN�m�P��)N&�uN��OPN)�N�a;O��O\�O�Nh�O�yOU�yO
	�N��N���P��Ni�.P%`�N���N)��O���P`�M���O��DNI3O%�NO.�=P"Q;Oɂ�O��N13�O��Od�OR�UM�k{O%�cOq�%N#,O��O��O�/SO
��N�/}Nf��O7��O`N�=�w<�9X<T��<D��;��
;�o:�o$�  $�  ��o�o�o�D���o�o�t��49X�D���D���D���D���e`B�u�u�u�u��o��t���t����
��1��j������������������/��/��/��/��`B��h�����o�o�o�+�C��C���P��P��w��w��w�,1�,1�,1�0 Ž49X�49X�<j�@��L�ͽT���aG��aG��q���y�#�����O߽�O߽���mmzz�����zmkmmmmmmmm��������������������#)6:B@76)-5ABCNWWNB65--------��������������������)2.*)#���

��������������������������������������������������������������������&)67BOOOKKB965)&&&&&R[hktv{yvtlh[QRRRRRR?HIUY_aggaUH@=??????
#&)+/62/#
��������������������������)2)&"�������-:B[gottjgB5)�����
�������*36<BFNKMHKEB6)$o�������������tipmjo-6COZhuyzui\C*��������������������sv���������������uqsy}��������������{xxy��� ).46,)��������������������������%;?:=HTmz{��{xmT;"%��������������������st���������������tss�� 
#/81/(#
�����PUbnpqnbUQPPPPPPPPPP������������������������)$���������������������������������./<HHQTH<70/........����������������������������������������#/7<ACHUWUTH></# #[anrz����zna[[[[[[[[##/1<A=</#fz���������������off����������������������
#0<LM@(&������������������������������������������������������������������������� ��������wz������zxwwwwwwwwwwot��������������tpmo35BFLKBBA75433333333bhlt������������th^b�����

��������Nan�����������}niUJNmns{�������������vpm+55@BNT[cgttlg[NB52+��
"
	����������!#'0BIUbnusnVI<0#!!��������������������ht����������th`Z[`dhrt}��������trrrrrrrr��������������������!)>BEHB=5)����������������������������������������U`g����������tmhgeTU����� *,�������HIU_bnpwzxnmb^USNHEH//1<HPSHH</.////////���������
"
������#/1<AHE<;/#OU`afmnpnja\UTOOOOOO�������������������������h�a�[�W�[�[�d�h�i�t�z�y�v�t�h�h�h�h�h�h�ɺú��������ɺֺ�����ֺɺɺɺɺɺɿ�������������������������������������������������������������������� ���������仞���������|�������������ûлһлû����������������������¼ʼ˼ʼ����������������|�u���������������Ľнٽݽ������ݽнĽ�������ŷŭŹ������������)�#��������/�(�$�#� �#�#�#�/�<�A�H�J�H�>�<�/�/�/�/�/�(�$�#�!�#�/�<�H�J�H�G�B�<�/�/�/�/�/�/������������������������������������N�D�B�A�@�B�N�[�c�t�t�g�[�NìáàÞàäìùþýù÷ììììììììƎƂƁ�~ƅƎƚƩƳ������������ƴƳƧƚƎ�z�f�Z�M�I�M�[�f�s��������������������z�{�u�s�w�{ŁŇőŠŭŸŭŤŠŘŔŏŇŁ�{�.�"��	��������	��"�.�;�G�L�R�M�G�;�.���������������ʾ׾�����������׾����"�	����ھѾо׾���	�"�)�.�1�6�6�.�"�����������	���"�"�"��	���������������h�[�B�)���)�6�B�O�[�hĀčėėčā�t�h�лû������������лܻ����	������ܻоf�Z�K�R�Z�g�s����������������������s�f�@�7�6�@�M�Y�f�r�x�x�r�f�Y�M�@�@�@�@�@�@�����N�5������N�s�������������������O�M�C�O�Z�[�]�f�h�k�h�[�O�O�O�O�O�O�O�O�ʼɼȼ��������������ʼּּּܼѼ̼μʼ��z�t�m�f�c�c�d�m�z�{�������������������z�-�*�&�,�-�:�F�F�;�:�-�-�-�-�-�-�-�-�-�-�0�,�#���
��������
����#�0�4�4�1�0���s�c�Z�Q�N�G�N�V�g�s�������������������ݿؿѿпѿֿݿ������(�5�(��������6�2�4�6�8�C�G�O�W�\�h�n�u�w�p�h�\�O�C�6����������������������������������������h�\�V�P�T�h�uƎƚ������������ƳƚƎƁ�h������������� ����$�+�0�1�/�$�!�����;�3�/�*�/�<�A�B�H�R�U�Z�a�b�i�f�a�U�H�;�U�T�I�H�H�H�H�U�Y�[�Z�U�U�U�U�U�U�U�U�U¿½¿¿������������������������¿¿¿¿����������ļĽ�����������0�'�'�"�
���������������������������������������������������~�y�|�����ݽ����'�����ʽ������b�^�`�b�o�{ǅǈǌǔǕǔǌǈ�{�o�b�b�b�b��
����$�,�$�!������������I�@�7�5�0�$��
��$�0�I�o�}ǂǂ�{�`�V�I��s������ļؼ���� �$�!�"����ʼ���H�>�;�;�;�H�T�X�\�T�H�H�H�H�H�H�H�H�H�HŔōŀ�v�yŇŔŠŭ������������������ŭŔ���������������	����������������������������������������������������������s�m�g�d�b�g�s�������������������������s�����������ùܹ����'�@�L�=����Թ������������~�r�[�`�~�������Ժ���ܺɺ������������������������������������������������������������������������������������������v�g�c�_�`�e�s�����������������������~�y�w�y�|�|���������������������������Z�U�U�Z�a�g�s���������������������s�g�Z�	���������	������	�	�	�	�	�	�	�	�����׾ξʾ��¾ʾ׾���������������h�[�M�M�O�UāĒĚĦīĦģğĞēčā�t�h���������������ĿɿĿ��������������������3�,�1�;�@�L�c�r�}�����������~�r�e�Y�L�3�H�/�'��������<�U�a�e�`�_�a�n�z�n�U�H�y�l�`�W�`�n�y�����Ľн����޽нĽ����y�лϻ̻лһٻܻ�������������ܻ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ùîìàÕàìùú����������ùùùùùùELECE7E0E7ECEPE\EiEuE�E�E�E�E�E�EuEiE\ELE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������������ĿƿĿ������������������� _ 3 2 U 5 _ R W = 9 @ H 2 % F 0 / � F 2 > c K 9 C R r G v X O i ; U  S Q h @ � Q L S L V 8 9 c D , V N ? f 9 E s R &   M L V ] X D Y @ 6 � T ) �    �  X  �  �  �  G  �  \  u  �  �  �  �  �  o  M    P  �  �    �  �  #  k  �  �  P  �  s  c    ?  �  G  �  ;    <  �  �  �  �  �  �  H  �  :    �  q  �  �  M  �  e  k  �  �  �  -  x  M  l  6  �  (  ;  {  �  �  7  �=C�<��㻣�
;�o:�o�49X���
��o������9X���
�o�49X�ě���C��o����1��9X�����#�
��1�T������H�9�ě������j��������`B��`B�m�h�8Q�C��t��m�h�8Q�0 ŽC���P��+�\)��C���P��P�����Q��w�T���Y��Y��ixսě���1�aG��49X�T���m�h��O߽@���o��E��Y���\)�ȴ9��hs��o������\)����\��{A���BC�B[}B��Bg�B��B$ �B�&B*y�B7�B��B�,B��B�:B7�Bx�B��B�B��B`FB0��B�B �BS�B5#B#LA�H�B1B�B#B'�B�nBYBD�B�bB�pBo�B"QBDBB�B�B��B]bB$��B�B�B��B-^�B ^B%�B��B�JBJ�BD^Bm�B!,B$>�B&��B��Bs2B
G
BW/BW�B+�B �B
��B�B'�:B��B�0B�
B�B!4A���BE'B@B�nB}�B��B$7�B>`B*�B��B��B-�B�B��BCXBGBz6BɣB�|B�B0E>B��B
�B@AB7�B"ǆA���B>B��B:�B'��BE�B�lB@qB�B0B@AB{�B<�B��B�Bw�B@TB%D`B<jB��B�B-H�B ;�B>NB� B�BDBɸB@]B�[B$�B&��B��B��B
<�BL�B?�B8�B ��B
��B��B'ټB��B@�B�yB�zB?RA�ΎA۪�@9�^Aq��A���@�\�@��A��A&��A���A§]A��A�$|A���A̹B�AD�sA��A`��AP�A[q|A�NA�:@�XAD�@َ�A�#VA�>@��A�b@yDA��A�KiA��Ba�A��B�B	JA�A��A�*[A��A�Y�A'�`B)]B	EiB��Ar�A��EA���AӟFA!�A���>�q7@�A���A�D�A�l~ArߒA�{�AZ�7AT�"AܧoAu�A?ޒ�A­�A",s@���C��mA�[�C�БC�,cAu��A��(A�G�@;�HAqA���@��@���A��wA%��A�A�~5AA�zjA���Á�BP�AC��A�Ab�AM��AZ��A��|A�ng@��AD��@��A���A�H�@��A��@u�xA逕A��"A�_BARA��eB��B	>VA�m�Aũ�A��sA�A�g�A"�B�uB	@�B�A�[A���A�uPAӅ�A!��A�p.=��@$�A���A��sA�9YAs�4A�W�AZ��AT0AܨkAv=?���A¹�A"͸@���C��-A�]�C���C�(OAu!s            	                           	      
         
               &   >   #   
   2      	            !      	                   
   &      &         "   :                  :   .                        +         ,               &                                    #                     #         #   #                  9                              #               )      /         !   1      !            1   %         !                        '   %                                                                     !         #                     1                              !               )      -            #                  )            !                           %                  Nn��NM(BN�ArNm�:N�N~N�5�N��zN"��N��O��|N�MN��N���N+��NJU�O�O��gN���N��O�!�O�ZoNf��O�-O2��O^_iN�m�PE��N&�uN>ywOPN)�N[(O%K�OI�	O�Nh�O��OU�yN�KEN��N���P��Ni�.P�[N���N)��O�O��MM���O�%NI3N�v�N�OO�æN���N13�O��Od�OR�UM�k{N�fO�5N#,O��N�KO�/SO
��N�/}Nf��O7��N��N�  �  $  \  ;  �  �  �  X  �  �  |  `  �  �  �    �  �  G  �  �  5  ;  
C  )  k  �  +    �  %  �  �  �    �  �  \  U  �  �  �  �  ]  �  q  ;  �  1  �    �  G  �  �  �  �  3  x  �  �  �  	"  �  �  �  ?  �  �  f  
T  �  �=�w<�9X<T��<D��;��
;�o:�o$�  �o�D���o�o��o��o�o��t��D���D���u�D����C��e`B���
��h��9X�u�ě���t����
���
��1�ě��t���/������������/�o��/��`B��h���o�o�o�@��'C�����P�'0 ŽL�ͽ0 Ž0 Ž,1�,1�0 Ž49X�49X�L�ͽu�L�ͽT������aG��q���y�#�����O߽�t�����mmzz�����zmkmmmmmmmm��������������������#)6:B@76)-5ABCNWWNB65--------��������������������)2.*)#���

��������������������������������������������������������������������&)67BOOOKKB965)&&&&&R[hktv{yvtlh[QRRRRRRAHLUX]aea_UHG?AAAAAA#%/)#����������������������� �����/;B[gnrrg[NB5)	�����
�������$)16BBEBBBFB=6))  $$o�������������tipmjo!*16CO\glmg\C6*��������������������|���������������|tx|����������������}|~���"),-.-)�������������������������"(;@DD@AHTnsx��mT;""������������������������������������������ 
#/81/(#
�����PUbnpqnbUQPPPPPPPPPP��������������������������������� ��������������������������./<HHQTH<70/........����������������������������������������#/<CE<5/&#[anrz����zna[[[[[[[[##/1<A=</#fz���������������off���������������������
#0<LM<%$���������������������������������������������������������������������������������wz������zxwwwwwwwwwwrt���������������yrr35BFLKBBA75433333333qt����������tmhlqqqq����

�������Tanz���������zpWQOPTv~�������������xsppv,57ABNQ[_fghg[NB53,,��
"
	����������!#'0BIUbnusnVI<0#!!��������������������ht����������th`Z[`dhrt}��������trrrrrrrr��������������������#),25665)	����������������������������������������rtv~�����������{trrr����� *,�������HIU_bnpwzxnmb^USNHEH//1<HPSHH</.////////���������
"
������!#'/<<EB<9/#OU`afmnpnja\UTOOOOOO�������������������������h�a�[�W�[�[�d�h�i�t�z�y�v�t�h�h�h�h�h�h�ɺú��������ɺֺ�����ֺɺɺɺɺɺɿ�������������������������������������������������������������������� ���������仞���������|�������������ûлһлû����������������������¼ʼ˼ʼ����������������|�u�нǽĽ��������������Ľнѽ۽ݽ�߽ݽн�������źŹŷ������������#���������/�(�$�#� �#�#�#�/�<�A�H�J�H�>�<�/�/�/�/�/�(�$�#�!�#�/�<�H�J�H�G�B�<�/�/�/�/�/�/������������������������������������g�e�b�g�t�t�g�g�g�g�g�g�g�g�g�gìáàÞàäìùþýù÷ììììììììƧƛƚƎƌƎƙƚƧƳƸ��������������ƳƧ��h�Z�M�J�O�\�f�s���������������������{�u�s�w�{ŁŇőŠŭŸŭŤŠŘŔŏŇŁ�{�"������"�.�.�/�;�A�G�M�H�G�;�.�"�"���������������ʾ׾�����������׾����	������޾־׾��	��"�(�/�1�1�+�"��	�����������	���"�"�"��	���������������O�-�"�"�)�6�B�O�U�h�vāčēčąā�t�[�O�û����������ûлܻ������������ܻлþZ�U�Z�`�f�s���������������������s�f�Z�@�7�6�@�M�Y�f�r�x�x�r�f�Y�M�@�@�@�@�@�@���������g�N�5����(�5�Z���������������O�M�C�O�Z�[�]�f�h�k�h�[�O�O�O�O�O�O�O�O�����������ʼҼּμʼ��������������������z�t�m�f�c�c�d�m�z�{�������������������z�-�*�&�,�-�:�F�F�;�:�-�-�-�-�-�-�-�-�-�-�
����
����#�/�#��
�
�
�
�
�
�
�
�s�m�g�b�]�]�g�s���������������������s�s�ݿӿѿѿؿݿ�����(�2�(�&���������6�2�4�6�8�C�G�O�W�\�h�n�u�w�p�h�\�O�C�6����������������������������������������u�h�\�Y�S�Y�h�uƎƚƳ��������ƳƚƎƁ�u������������� ����$�+�0�1�/�$�!�����H�@�=�A�H�U�[�a�d�a�`�U�H�H�H�H�H�H�H�H�U�T�I�H�H�H�H�U�Y�[�Z�U�U�U�U�U�U�U�U�U¿½¿¿������������������������¿¿¿¿����������ļĽ�����������0�'�'�"�
�������������������������������������������������~�z�|�����ܽ����%������ǽ������b�^�`�b�o�{ǅǈǌǔǕǔǌǈ�{�o�b�b�b�b��
����$�,�$�!������������V�J�=�9�0�.�0�8�=�I�L�V�b�o�r�w�q�o�b�V���������ɼ޼������������ּʼ��H�>�;�;�;�H�T�X�\�T�H�H�H�H�H�H�H�H�H�HŠŔŌņ�|łŔŠŭŹ����������������ŹŠ���������������	����������������������������������������������������������s�i�g�e�g�k�s�~���������������������u�s�������������ùܹ���+�$���Ϲù������������~�t�r�w�~�������ͺ��ٺɺ����������������������������������������������������������������������������������������������v�g�c�_�`�e�s�����������������������~�y�w�y�|�|���������������������������Z�U�U�Z�a�g�s���������������������s�g�Z�	���������	������	�	�	�	�	�	�	�	�׾վʾžƾʾ׾����������׾׾׾׾׾��[�W�V�[�a�h�t�~āčĎĕĕčĈā�t�h�[�[���������������ĿɿĿ��������������������3�,�1�;�@�L�c�r�}�����������~�r�e�Y�L�3�/�*�#������#�/�<�H�I�R�K�H�?�<�/�/�y�l�`�W�`�n�y�����Ľн����޽нĽ����y�лϻ̻лһٻܻ�������������ܻ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ùîìàÕàìùú����������ùùùùùùELECE7E0E7ECEPE\EiEuE�E�E�E�E�E�EuEiE\ELE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������������ĿƿĿ������������������� _ 3 2 U 5 _ R W * 6 @ H - ( F - / � 3 2 4 c L % E R } G < X O d 0 T  S N h . � Q L S N V 8 @ B D ) V < ' a + 9 s R &   M 6 7 ] X " Y @ 6 � T " �    �  X  �  �  �  G  �  \  �  d  �  �  �  A  o  <  �  P  �  �  N  �  e  {  �  �  �  P  ^  s  c  z  Z  �  G  �  �    �  �  �  �  �  �  �  H  =      8  q  �    �  t  1  k  �  �  �  -  �  -  l  6    (  ;  {  �  �    �  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  �  �  �  �  �  �  �  �  z  k  \  M  9       �  �  �  �  �  $  #  "  !  !                     �  �  �  �  x  Y  \  D  +    �  �  �  �  �  ~  \  3  �  �  ^    �  d  	  �  ;  5  .  &        �  �  �  �  �  �  v  V  5    �  �  �  �  �  �  �  �  �  �  �  �  �  {  k  Z  J  :  ,         �  �  �  �  �  �  �  �  �  n  L  )  �  �  {  T  +     �  �  v  �    t  i  \  O  A  3  &    	  �  �  �  �  �  �  �  b  C  X  I  :  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  �  9  K  W  c  l  w  �  �  �  e  B    �  �  �  U    �  �  {  �  �  �  �  �  �  r  T  5    �  �  �  s  6  �  �  g  �  �  |  z  x  v  t  r  p  h  ^  T  I  ?  5  (       �   �   �   �  `  Y  S  L  D  <  3  )        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  Z  5    �  �  �  �  a  �  �  �  t  c  _  |  �  �  �  �        �  �  �  �  �  >  �  �  �  �  �  x  Z  9    �  �  �  p  0  �  �  D  �  �     �  �  �  �  �    
        	  �  �  �  y  N  #     �  �  X  �  �  �  v  h  S  @  /      �  �  �  n  o  L  "  �  �  �  �  �  �  �  �  �  g  H  (     �  �  �  s  U  8      �  �  �        &  :  G  E  ?  5  -  (         �  �  �  �  �  �  �  �  z  k  W  D  2  #    �  �  �  �  �  {  c  I  -    |  �  �  �  �  �  �  u  h  U  ?  %    �  �  �  >  �  Z   �  5  /  )  "         �  �  �  �  �  �  f  ?    �  �  �  l  �    6  ;  3      
  �  �  �  @  �  �  5  �  Z  �    �  	�  	�  
  
1  
@  
;  
  	�  	�  	�  	B  �  �    W  �  �  s    �  �    !  '  )  %        �  �  �  z  M  6    �  u  �  �  k  d  ]  T  K  A  6  *      �  �  �  �  �  �  �  )  E  a  �  �  �  �  �  �  �  �  {  T  +      �  �  j    �  �  �  +  "      
         �        	  	  	    �  �  �  w  �  �                   �  �  �  �  �  �  �  �  	  ,  �  �  �  �  �  s  c  T  D  3       �  �  �  �  Z    �  k  %  $  #           �  �  �  �  �  �  �  �  �  |  f  N  6  �  �  �  �  �  �  �  �  �  h  M  2     �   �   �   �   �   g   H  G  w  �  �  �  �  �  �  �  �  �  j  8  �  �  T  �  �  ;  z  �  �  �  �  q  X  >       �  �  �  [  '  �  �  _    �  �    �  �  �  �  �  �  �  �  �  v  b  N  <  )      �  �  �  �  �  }  |  w  ^  F  ,    �  �  �  �  n  N  .      �  �  �  �  �  �  ~  m  V  8    �  �  r    �  d  S  �  �  �  �  \  Z  T  H  8  &    �  �  �  �  ^  ,  �  �  �  3  �  K  �  �  �    <  N  U  U  S  O  G  <  *    �  �  �  �  �  U    �  �  �    �  �  �  �  �  x  \  A  )    �  �  �  �  �  e  �  �  �  �  |  ^  @  �  �  p  F    �  �    J    �  �  �  �  �  �  �  v  K    �  �  y  *  �  g  @    �  �  �  >  �  �  �  �  �  �  r  W  <       �  �  �  �  �  �  e  F  &    [  [  U  B  $      �  �  �    +    �  �  �  C  �  �  k  �  �  �  �  �  �  �  �  �  �  �  �  �  q  `  N  6      �  q  s  u  w  x  x  x  x  u  n  g  `  R  @  .      �  �  �  �      (  *      %  2  :  %  	  �  �  �  T  $  �  X  �  a  �  �  �  �  �  �  �  �  k  :  �  �  <  �  Y  �  g  �  �  1  3  5  7  7  +         �  �  �  �  �  |  b  C  "    �  �  �  �  �  �  �  �  �  �  �  �  m  H     �  �  �  ?  �  ^       �  �  �  a  ,  �  �  ^    �  o    �  b     �  5  �  �  �  �  �  �  �  �  �  �  �  �  s  ^  C    �  �  a  �  |  �  /  ?  F  G  @  1      �  �  �  c  ,  �  �  }  :  �  k  �  �  �  �  �  �  �  �  f    �  5  �  �  �  \  �  �  �    x  �  �  �  �    S    �  �  k  .  �  �  ~  >  �  `  u     n  �  �  w  X  1    �  �  �  r  Z  E  5  (         -  A  �  �  �  �  �  �  �  �  �  �  �  z  r  j  b  Z  R  J  A  9  3  &    	  �  �  �  �  �  �  �  �    r  b  P  *   �   �   �  x  i  [  M  >  /      �  �  �  �  �  d  ?    �  �  �  �  �  �  �  �  �  �  �  �  �  o  N  (     �  �  g  ,  �  j  �  �  �  �  �  �  �  z  d  G  +    �  �  �  �  y  Z  ;     �  �  �  �  �  �  �  �  �  z  h  [  N  1    �  �  y  D    �  ,  �  �  	  	  	  	!  	  	  �  �  �  N  �  w  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  j  Z  H  5  #    �  �  �  �  �  t  ]  B  +    �  �  �  �  S    �  �  �  m  �    �  �  �  1  T  p  �  �  �  �  �  �  R  �  �    z  �  �  x  ?  3  &      �  �  �  �  �  q  Z  H  0       &  �  \   �  �  �  �  �  �  �  �  �  �  �  �  z  p  h  a  Z  L  =  -    �  �  �  �  `  I  0    �  �  �  �  Q    �  �  _    �  �  f  Z  O  C  8  -  "      %  4  B  C  <  4  -    �  �  �  
T  
M  
<  
  	�  	�  	�  	T  	  �  �  1  �  �  X  �  �  2  �  �  �  �  �  �  �  �  �  s  R  +  �  �  �  G  �  �  M  �  �  �  �  �  �  �  �  �  �  �  n  W  =       �  �  �  �  s  X  =