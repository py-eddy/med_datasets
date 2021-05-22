CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�?|�hs     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�H�   max       P���     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�P     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @F�z�G�       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vrfffff       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q�           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��          4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �J   max       <�/     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�%�   max       B4=�     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4�     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��a   max       C�gl     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��   max       C�g     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          S     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�H�   max       P�Ǧ     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?������     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =�P     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @F�z�G�       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vrfffff       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q�           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�o�         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���vȴ:   max       ?ϝ�-V       cx   
               <            	         !   K               7         
   
   S      &            
         (   (                  !      $                     
      9   
   	         #                     H            !               
               Ny^�M�H�O�k�O2l�O|��PY2uN�%N�aN{�N�K�N�.PG�O8HFPe@�P��N��;N�iN��O�PN]��N�t�N��	N��BP���PT�O�aNw\�N�sN���O	�N]k|O׆"P=].P	OF �N�a,O+*7NN��N"��O��jN.�iOo��N�_Oi�O	!O�EO�sVN���N�m�O�n�P��N|�kN��NZ��N��jOL�1NG�Op	pN}�yN�gO�^�O�P�֩N8pN��O3��O���N۴�N���O���O
^ENR��N��OA��O66
N5Y O3vI=�P<�j<T��<D��%   %   ��o�o��o��o���
���
���
��`B�t��t��#�
�49X�49X�D���T���u��C���t���t����㼛�㼛�㼛�㼛�㼣�
���
���
��9X���ͼ������o�+�C��t��t���P����w��w�#�
�'0 Ž0 Ž8Q�8Q�8Q�<j�@��H�9�L�ͽP�`�T���T���e`B�e`B�ixսq���u�u�}�}�}󶽁%��%��\)�������
���
���
#%%#
rz�������zrrrrrrrrrr��������������������lmtz������������zoml��)*'&	�������)B[p����zsh[B-������� �����������������������������������


	���������?BO[ghtwtmh[OIBBB;??9<DHUaca_ZUTHD<59999HUgn���������~nh[K?HMOT[hks������th[SNMMu������������~vru!)6BO^imj\O6

���

����������
!#/62/#
	#%/<>B</#��
#,6874,!
������!*6BCIEC@6*#/8<HJKLMH<60/*#������������������������������������������)Bg��uRDB5'%���������������	������]bm�������������re]]����������������������������������������#%./<?<3/#"su����������������us��������������������#0IXin{�~{kWSI<0$z}��������������{wz!)<BNgt�����g[N5&!W[bhmt��������th[ZQW������������������������������������������ ������������##&/<<</##/:;?HNPONLH;/"#�����������������������
#'%$ %'!
�����������������������NOT[ht�������th[OFEN��������������mz�������������zlffm��������������������05BNXXTNB55/00000000#00<HIOLI<0/%#7;HTahmolaZTH;72/027IUanz��������ztnUD@I")*5BJFB:5) """"""""xz~����������zprxxxx&),6;<;B6))"&&&&&&&&��������������������2;DJO[hiqwqjh^[OB=62aamqvrnmka^]aaaaaaaa0CILUVY\^aeebUI?82/0��������������������!#cg������������~og\`c�������������������������5[_YB8)�������������������������#&!	�������������������������������! ������������������������������������������������������������������

���������������������������������������ehntw�����������tkge������������������������

������������TV]anrz�������znaXUTE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ݽڽٽٽݽ����޽ݽݽݽݽݽݽݽݽݽݾ���f�T�M�G�J�f�q��������þ;̾�������àÜÓÐÒÉÂÃÇÓàèìïúûùìçà£¦²¿��������������¿²¦���������9�G�T�m�y�����{�m�G�	�����	����������������	���	�	�	�	�	�	�	�	��������'�'�'�����������V�M�I�>�I�V�b�o�q�o�b�\�V�V�V�V�V�V�V�V�x�v�u�w�x�x�������������������������x�x�����������������������������������������2�$�����	��"�H�T�a�m�����������m�H�2�Y�R�L�I�I�L�Y�e�r�w�~�����������~�r�e�Y�ֺ����x�a�~�����ɺֺ����(�,�!�����/�����������"�;�H�T�a�g�w���j�a�T�H�/��������������������������������������������������������������������������������ùîìæëìùúü��ùùùùùùùùùù�л��������������лܻ��������ܻпG�A�;�3�3�;�;�G�T�U�T�P�M�I�G�G�G�G�G�G��������������
��#�/�3�3�/�#�"��	�����	������������� �	����"�%�)�"��	�	��ھ׾Ѿо׾߾��������������������\�T�_���������"�T�n�u�a�/���������������r�m�c�^�g�������������������������<�0����	��#�<�I�U�b�v�|�y�n�b�U�I�<�)�(�$�)�6�B�O�O�O�N�B�6�)�)�)�)�)�)�)�)������������%� ���������¿µ»¿������������������������¿¿¿¿�������������������ʾ׾�޾ھ׾Ӿʾʾ����ʼɼǼʼҼּ�����ּʼʼʼʼʼʼʼ����{�p�a�Z�N�C�8�N�s�����������������������ݿ����m�]�Q�T�m�����Ŀ���	�(�1�4�������ʾžžھ޾���	��(�1�2�1�)�"��	�������s�g�Z�O�F�F�N�Y�g�s�v���������������@�:�3�'��%�'�3�@�L�Y�e�r�v�w�r�e�Y�L�@���������������������������������������������������������������������������[�T�O�B�B�B�E�L�O�W�X�Y�[�_�[�[�[�[�[�[������������������������������������������������������������������������������E*EED�D�D�D�D�D�EEEE*E7ECEGELELEDE*�Ŀ����Ŀʿѿݿ������������ݿѿĽ����������������������Ľ̽ڽ�ܽսнĽ���	�	�����������	���"�)�.�1�.�"�ƮơƚƋƁ�u�e�^�u�zƎƧƵ������������Ʈ��ŹůšŨŹ��������*�6�C�Y�O�E�6������5�0�0�3�5�B�N�S�T�O�N�B�5�5�5�5�5�5�5�5�������!�!�-�:�>�B�:�:�-�*�!�����
������(�5�Q�Z�c�i�U�N�A�5�(�����������~����������'�2�3�����ݹù����I�=�=�:�;�=�I�M�V�Z�b�V�I�I�I�I�I�I�I�I�N�J�A�5�3�0�5�A�N�Z�`�g�m�o�g�Z�N�N�N�N������ÿ��������������������������������ƳƪƧƚƚƚƢƧƳ��������������ƳƳƳƳ�ܹù��������������ùϹܹ���������;�;�4�;�H�T�a�e�a�\�T�H�;�;�;�;�;�;�;�;���ӽнŽнݽ����4�A�L�Q�B�4�(��������������������������������������������h�^�c�h�tāĆā�|�t�h�h�h�h�h�h�h�h�h�hŇł�b�_�X�b�eŔŠŭŵ��������ŹŬŠŔŇ�h�a�[�P�L�B�O�R�[�h�t�w�t�q�t���t�s�h���ùìâÙ×Ð�tàù������)�D�E�M�B����������������������������������������޽!��������!�(�.�6�:�A�:�.�!�!�!�!����������!�+�.�:�G�H�H�?�:�.�"�����������y�S�W�e�y�����нؽؽ۽ֽͽ������������������������$�&�'�$������������'�,�4�@�M�T�M�E�@�4�-�'���Y�W�f�������ȼ���������ּ����f�Y�����������'�(�4�>�4�'�%��������Y�T�M�B�M�X�Y�_�f�r�l�f�Y�Y�Y�Y�Y�Y�Y�Y¿³²©¦¥¦²¼¿������������¿¿¿¿�H�<�/�&�#��� �#�/�H�U�W�a�i�a�Y�U�O�H���������������������Ŀ˿ǿſĿ����������ѿοпѿؿݿ�����ݿӿѿѿѿѿѿѿѿѾ��׾Ѿ̾ʾȾľʾ׾������ ������� h �  [ D 0 h ? l ^ b N A P 4 G T v 4 � d � - S N 0 ) 9 x c 6 k e 7 l o N $ n Y 9 Z R A U a { 6 r / a R X N + 7 I b L U 4 8 b 4 e � 7 = i � ; : R , D q :    �  s    �  !  �  [  9  R    �  �  �  H  [  �  �  �  "  �  9  "    �       |  �  �  o  u  Q  �  {  �  J  �  _  �  �  L  	    �  <  �  �  �    &  <  �  �  j  �  �  u  W  �  8  �  D  �  *  �  �  �  	  �  �  :  Q  �  �  �    �<�/<���t��o��t��u�D���ě��o�D���o�ě���㽧�`B��o�T�����㽇+�u���ͼ��ͼ������ͽ#�
�ixռ�󶼼j��h��`B���ͽ\)�q���}�0 Ž�P�]/��P�\)�����w��t��,1�u�H�9�aG��}�@��P�`��C����ͽ]/�Y��ixսY���e`B��hs�m�h�ixս�hs��C��J�y�#��\)��9X��j��t���C���^5���罕����P���`��
=��9X���mBE�B�B!rB ��Bd�B�BBKB��B�Bm�B�7B�B��B�$B#��B_yBaB$? B/ԌB��B!�B��B2�B��B �B�B�B��B4=�B ��B&�@B*}B��BN=B"��Bn�BFB�@A�%�BBB�kB�B�CB�B ��BQB�>B%��A��BR�B/�BłB�VBK]B TA�<�B'�B�tB>B
��B�qB�SB.�BA'B�LBh B��B)��B-�BJ�B7eB
غB
qFB'�B@-BUBAKB<�B!3aB ��B:�BA\B?	BM�B�&B�B��BA�B�B>�BɃB#��B��B�mB$?�B0BB�3B!�kB�~B?�B7�B �YB=�B;BCrB4�B ��B&�kB*��B	0#B@�B"CBL�B�BĜA���B>�B��B?�B?"B��BL2B?�B��B&?-A��B�
B�TB��BAuBA�B>�A�v`B&��BcB>�B
��BJBp<B?�B?�B�OB�fB'NB)�pB-��BCtB?�B
�B
�B��B�nB��C�glA+��AF݌A�+�A�a�Aa|�A�D�?}��B��@�R�A���A�eS?�:�@C�A���@�`�A�jA�t�@��FAeʷA�f�A�?1AVEA�ܭA��A�ùA��A�<�A���AO��A�+A���Aw`>AY�sA���?ǡ1A��-AJ'A�!$A��s@�c�C�r�A}wA$rYA[��B�A���A�a,@nV�A�e�>���BM4A��EA�UBD8>��aA�մA3�:A��A�RdA�q>A��A���A�)'A�AA7A �uB�:@˂�@��K@�^�@���A���Aî4At�/A}9KAUm�C�gA,�EAG!A˵�A��Ab�	A��?~�B�W@�K�A�{A���?���@DQA�f@�)A�~�A�4@�+Ad�fA��A�}�AV2A�trA��A�1A�}oA�'�A���AOW�AA�e8Aq6�AZ�A���?�$:A�w�AJ�A�~�A��+@�JC�y`A{vA"�A\ [B�A��{A�;@s��A��=��BjA��[A�{�B?�>C�aA��sA4��A��A܉OA�~FA�~�A�u/A�K$A�3A~AMbB��@ʟ�A��@���@��A�[�A�v�As�A~�]AUD                  =            	         "   L            	   8            
   S      '            
         (   )                  !      %                           :   
   	         $                     H            !                           	            '         1                  -      ;   )            %               I   +   #                  +   5   %                                    #   '            +                     !               ;            %         3                                       %                  #      '                              ?   +   #                  +   1   !                                    #   #            +                                    5            %         3                     Ny^�M�H�OÑ�O2l�N�>lO�v�N�%N�aN{�N�EN�.O���N��*P �0O���N�$N�iN��O*��N]��Nh�`N���NƃP�ǦPT�O�aNw\�N�sN���O	�N]k|O׆"P#U'O��O"v2N�a,O~�NN��N"��O<�{N.�iOo��N�_OL�O	!O�EO��N`�N��'O�c(P��N|�kN��NZ��N��jN���NG�O[A`N}�yN�gO�� O�P`��N8pN��O'��O���N۴�N���O���O
^ENR��N��N��7O66
N5Y O3vI  V    �  2  �  �   �  G  �  �  �  �  H  �  �  �    �  �  .  �  �  �  �  �  3  S    !  6  �  �  �  -  '  �    d  �  �    �    �  �  =  �  �  �  �  b  �  �  �  d  Q  �  N         J  	�  T  Z     �  �  6  �  *    �  �  �  ~  �=�P<�j<#�
<D���o��C���o�o��o���
���
�#�
�e`B��`B�T���#�
�#�
�49X��P�D����o��o��t���h��t����㼛�㼛�㼛�㼛�㼣�
���
�ě�����/���+���o�,1�C��t��t���w����w�'',1�49X�49X�8Q�8Q�8Q�<j��%�H�9�P�`�P�`�T���]/�e`B�y�#�ixսq���y�#�u�}�}�}󶽁%��%��\)��1���
���
���
#%%#
rz�������zrrrrrrrrrr��������������������lmtz������������zoml���	 ������� +6BO[gpzvlh[B61!������� �����������������������������������


	���������@BOZ[hskh[OLB<@@@@@@9<DHUaca_ZUTHD<59999FIUanz������znaXOLGFU[hltz����th[XSRUUUU����������������}}�)6BW]`eb[OB6)���

����������
!#/62/#
	#%/<>B</#��
"#&&#"
������!*6BCIEC@6*)/1<?GGHIH<<4/,)))))������������������������������������������)5Bmzzt[LB5,)���������������	������]bm�������������re]]����������������������������������������#%./<?<3/#"su����������������us��������������������#0IXin{�~{kWSI<0$��������������~|����(2BNVgt�����tg[NB5+([[`dot~��������th[W[������������������������������������������ ������������##&/<<</#!%)/2;AHJKLKH;/&"  !�����������������������
#'%$ %'!
�����������������������KOPV[hjt�������th[KK��������������mz�������������zlffm��������������������25BMNUPNB75022222222#,0<FINKI<10'#8;HTamnkaYTH@;830038IUanz��������zunUEAI")*5BJFB:5) """"""""xz~����������zprxxxx&),6;<;B6))"&&&&&&&&��������������������@BFOT[ahmhh[XOGB@@@@aamqvrnmka^]aaaaaaaaEIUX[]`addb\UI@9301E��������������������!#f��������������rjgbf�����������������������)BN[UB6)��������������������������#&!	�������������������������������! ������������������������������������������������������������������

���������������������������������������otu�����������tqoooo������������������������

������������TV]anrz�������znaXUTE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ݽڽٽٽݽ����޽ݽݽݽݽݽݽݽݽݽݾs�f�Y�O�Q�s����������ǾǾ�����������sàÜÓÐÒÉÂÃÇÓàèìïúûùìçà²©¦ ¦¦²¿��������������¿²²²²��	����������	��.�G�`�h�p�n�g�`�G�.��	����������������	���	�	�	�	�	�	�	�	��������'�'�'�����������V�M�I�>�I�V�b�o�q�o�b�\�V�V�V�V�V�V�V�V�x�v�v�x�y�z�����������������x�x�x�x�x�x�����������������������������������������H�;�/�)���%�/�H�a�m�z�������z�m�a�T�H�Y�R�R�Y�`�e�o�r�~���������~�r�e�Y�Y�Y�Y�ֺɺ��������ֺ����� ����������������"�/�;�H�T�[�b�j�p�u�v�a�T�;�/���������������������������������������������������������������������������������ùîìæëìùúü��ùùùùùùùùùù�л����������ûлػܻ������������ܻпG�A�;�3�3�;�;�G�T�U�T�P�M�I�G�G�G�G�G�G��������
��#�*�/�0�/�/�#���
���������	��� ��	����"�$�(�"��	�	�	�	�	�	��ܾ׾ӾҾ׾����������������������e�`�b�k���������	�"�E�R�;�"�	�������������r�m�c�^�g�������������������������<�0����	��#�<�I�U�b�v�|�y�n�b�U�I�<�)�(�$�)�6�B�O�O�O�N�B�6�)�)�)�)�)�)�)�)������������%� ���������¿µ»¿������������������������¿¿¿¿�������������������ʾ׾�޾ھ׾Ӿʾʾ����ʼɼǼʼҼּ�����ּʼʼʼʼʼʼʼ����{�p�a�Z�N�C�8�N�s�����������������������y�m�_�S�Z�m�����Ŀ����!���ݿĿ����	��վѾ׾�����	���&�*�,�)�$�"��	�������s�Z�R�N�K�K�N�Z�g�s�t�������������@�:�3�'��%�'�3�@�L�Y�e�r�v�w�r�e�Y�L�@����������������������������������������������������������������������������[�T�O�B�B�B�E�L�O�W�X�Y�[�_�[�[�[�[�[�[���������������������������������������˻���������������������������������������E*EED�D�D�D�D�D�EEEE*E7ECEGELELEDE*�Ŀ����Ŀʿѿݿ������������ݿѿĽĽ������������������������Ľɽֽ޽ڽнĿ�	�	�����������	���"�)�.�1�.�"�ƮơƚƋƁ�u�e�^�u�zƎƧƵ������������Ʈ��ŹűţŪŹ�����������*�0�<�*��������5�2�1�5�7�B�N�P�R�N�M�B�5�5�5�5�5�5�5�5������!�#�-�:�=�@�:�9�-�(�!�����������(�5�P�Z�b�g�Z�S�N�A�5�(�����������~����������'�/�2�����ܹù����I�=�=�:�;�=�I�M�V�Z�b�V�I�I�I�I�I�I�I�I�N�J�A�5�3�0�5�A�N�Z�`�g�m�o�g�Z�N�N�N�N������ÿ��������������������������������ƳƪƧƚƚƚƢƧƳ��������������ƳƳƳƳ�Ϲʹù����������ùϹϹ۹ܹ߹ܹ׹ϹϹϹ��;�;�4�;�H�T�a�e�a�\�T�H�;�;�;�;�;�;�;�;��ڽѽݽ�����4�A�J�M�O�@�4�(����������������������������������������������h�^�c�h�tāĆā�|�t�h�h�h�h�h�h�h�h�h�hŇ�n�b�^�b�n�zŇŔŠŭű������ŹŭŦŠŇ�h�a�[�P�L�B�O�R�[�h�t�w�t�q�t���t�s�h��ìåÜØÌÊàù������)�6�>�?�6��������������������������������������������޽!��������!�(�.�6�:�A�:�.�!�!�!�!�����������!�,�.�:�F�G�>�:�.�"�����������y�S�W�e�y�����нؽؽ۽ֽͽ������������������������$�&�'�$������������'�,�4�@�M�T�M�E�@�4�-�'���Y�W�f�������ȼ���������ּ����f�Y�����������'�(�4�>�4�'�%��������Y�T�M�B�M�X�Y�_�f�r�l�f�Y�Y�Y�Y�Y�Y�Y�Y¿³²©¦¥¦²¼¿������������¿¿¿¿�<�1�/�#�%�/�<�E�H�I�U�Y�U�U�K�H�<�<�<�<���������������������Ŀ˿ǿſĿ����������ѿοпѿؿݿ�����ݿӿѿѿѿѿѿѿѿѾ��׾Ѿ̾ʾȾľʾ׾������ ������� h � ! [ /  h ? l l b 6 8 < - L T v 0 � j l / B N 0 ) 9 x c 6 k ^ 6 e o M $ n K 9 Z R A U a o 6 l , ` R X N + : I Y L U - 8 \ 4 e � 7 = i � ; : R  D q :    �  s  �  �  �  +  [  9  R  �  �  �    y  �  �  �  �  w  �  �  �  �  �       |  �  �  o  u  Q  <  �  �  J  y  _  �  �  L  	    �  <  �  �  z  �       �  �  j  �  �  u    �  8  b  D  *  *  �  �  �  	  �  �  :  Q  �  �  �    �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  V  K  @  5  *         �  �  �  �  �  q  U  9    �  _  �                       �  �  �  �  �  �  W    �  �  �  �  �  �  �  �  �  �  r  R  1    �  �  �  �  �  �  E   �  2  #    �  �  �  �  e  8    �  �  |  P  7  4    �  j  �  �  �  �  �  �  �  �  �  �  �  �  �  t  L    �  �  k  "  �  �  v  �  �  �  �  �  �  �  �  �  �  X    �      �  B  v   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  G  A  <  6  0  +  &  !              �  �  �  �  �  �  �  �  �  �  �  �  }  s  i  ^  T  K  A  7  -  $        �  Z  q  �  �  �  |  l  \  D  +    �  �  �  �  |  [  L  \  m  �  �  �  �  �  �  �  �  �  �  �  x  m  _  J  5        �   �  �  �  �  �  �  �  �  �  �  �  �  r  O  ,    �  �  �  z  (  �    -  :  B  G  F  A  5  #    �  �  E  �  b  �  9  �  �    2  �  �  �  �  �  �  �  �  �  s  2  �  �  2  �       �  �  �  �  �  �  �  �  �  �  �  k  L  %  �  �  �  n  T  ;   �    �  �  �  }  v  m  b  V  J  >  3  .  )  %  $  "                �  �  �  �  �  �  �  �  �  �  |  h  T  @  ,    �  �  �  �  �  �  �  �  j  O  3    �  �  �  �  t  Q  .  
  �  �  3  Y  h  p  t  x  �  �  }  c  -  �  �  0  �  �  �  b  .  )  %  !             �   �   �   �   �   �   �   �   �   �   �  L  N  Z  u  �  �  �  m  T  >  C  e  U  @  )      �  �  �  e  �  �  �  r  V  ;       �  �  �  �  �  u  ^  7    �  �  �  �  �  �  �  �  �  �  �  �  v  a  L  ;  3  +  #        �  �  �  �  �  �  �  T    �  �  )  �  �  :  �  b  �  -  �  �  �  �  �  p  g  m  r  d  a  _  W  ?        �  �  $  �  3    �  �  �  �  �  e  �  s  Q  (  �  �  r  !  �    <   �  S  =  '    �  �  �  �  �  V  6  !    �  �  �  �  ]  7        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  !  
  �  �  �  �  q  O  1    �  �  q  ,  �  �  Q    �  e  6  /  (  )  +  *  (  $      
  �  �  �  �  �    U  <  #  �  �  �  �  �  �  �  �  �  �  w  n  b  V  I  =    �  �  �  �  �  ~  t  g  T  >  &      �  �  �  �  �  u  \  <     �  �  �  �  �  x  V  9  &    �  �  �  z  L    �  �  b    4  �  �    %  ,  )  "      	  �  �  �  �  c    �     �  d    !  &  &  '  #      �  �  �  �  �  v  C    �  �  �  d  �  �  |  p  e  Z  Q  I  A  ;  5  /  )  #          
    �  �  
  �  �  �  �  �  �  �  l  5  �  �  |  *  �  ~  �    d  V  H  9  &    �  �  �  �  �  �  �  p  ]  K  =  D  K  R  �  �  �  �  u  a  L  0    �  �  �  �  y  q  h  `  W  N  F  A  c  }  �  �  �  �  �  �  b  5  �  �  ]    �  ]  �  h  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  `  O  >  �  �  �  k  5  �  �  L  �  �  p    �  �  \  �  �  '  �  +    v  n  e  W  I  :  (       �  �  �  �  �    k  M  /    �  �  �  �  �  q  ^  I  3    �  �  �  M    �  X  �  R   �  �  �  �  �  �  �  �  �  f  H  (    �  �  �  �  [  7    �  =  )      �  �  �  �  p  F    �  �  �  �  u  F    �  J  �  �  �  �  �  �  �  p  E  2    -  �  �  �  F  �  �    h  �  �  �  �  �  �  �  �  �  �  �  �  f  9    �  �  �  V  &  �  �  �  �  �  �  �  �  x  ]  =    �  �  �  9  �  �  i  !  �  �  �  �  �  �  c  A    �  �  �  d  )  �  �  1  �  `  �  S  \  G    �  �  �  �  �  �  �  F  �  �  -  �    l  �  Y  �  �  �  �  �  �  �  e  D  !  �  �  �  �  c  @           �  �  �  �  �  �  �  �  �  �  w  Y  :    �  �  �  j  ?    �  �  �  �  �  �  �  �  �  �  �  |  W  2    �  �  �  _  0  d  P  <  )      �  �  �  �  �  �  �  s  h  g  f  R  7    �    �  �  �    -  B  N  N  =    �  �  m  /  �  �  �  �  �  �  �  �  �  �    o  _  H  1    �  �  �  Q    �  �  w  D  L  6      
  	  �  �  �  �  ^  +  �  �  �  J    �  ~      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  j  Y     �  �  �  �  �  �  �  �  �  b  C    �  �  �  t  =     �  �          �  �  �  �  �  t  K    �  �  w  B  �  �  �  J  ,      �  �  �  �  n  R  7    )  #  �  �  �  �  �  �  	8  	�  	�  	u  	6  �  �  ]     �  �  m  '  �  g  �  1  ]  J    T  L  D  <  4  +        �  �  �  �  �  �  �  y  [  =    Z  J  :  %    �  �  �  �  x  P  /    �  �  �  =  �  a   �  �  �  �  �  �  �  �  �  e  6  
    �  <  �  +  �  �      �  �  �  �  �  }  b  H  0     .  $    �  �  h  �  `  �  i  �  �  �  x  m  b  W  K  <  %    �  �  �  �  �  |  w  n  d  6  "    �  �  �  �  �  �  �  �  w  a  J  3       �   �   �  �  �  �  W  6    �  �  T      �  �  {  �  P    �  �  E  *  (  '        �  �  �  �  �  \  '  �  �  `    �  �  %         �  �  �  �  �  y  M    �  �  X    �  �  |  B    �  �  �  {  q  e  W  H  9  *      
     �  �  �  �  �  �  2  L  n  �  �  �  �  �  �  �  i  =    �  �  i  3    ,  u  �  g  ^  X  /    �  �  n  6  �  �  x  *  �  t    �  |  8  ~  ~  ~  o  W  >  %    �  �  �  �  f  �  u  )  �  �  j  )  �  �  f  ,  �  �  �  \  %  �  �  n  -  �  �  K  �  z  d  w