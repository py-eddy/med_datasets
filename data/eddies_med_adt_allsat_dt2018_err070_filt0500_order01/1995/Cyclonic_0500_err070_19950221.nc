CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�^5?|�     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�)   max       P�V�     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��-   max       <ě�     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F�\(�       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @vtQ��       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @R            �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �)��   max       <�t�     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4�     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��N   max       B4�E     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?X   max       C�o"     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?#M�   max       C�j;     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          g     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�d�   max       P��P     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��J�M   max       ?�-�qv     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��E�   max       <ě�     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F���R       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @vtQ��       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @R            �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�u�         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @g   max         @g     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?�*�0��       cx                        g      $   1   )         0   
                              $   
   ,   ,   P                           ^         1            S                        	   2   ,      
         '   1               
               A         S      N�T;O�IM�)OE��N��}OP�N��cP�V�No��Og�.P ]_O���O���Num�P7r�N��HO�tmOM�VNkq�N)�/M�
;N�|OP��N%��N���Oѥ�N��O�h�OʘOɋ�O�"�O��8OfrbO��ZO!N��_N�}'N��zO���N(ȳN��O��N��{O�P$cP.`SN�N.�N�BYP��PN��O�^�P�NX��O�f�P	��O�%N��9N!�N�OZ+�O�ޚN���M�O?Ov�tN�R�N^b�O�OcORx�O,bO�UEN��N���Okm�N��Oʯ<ě�<D��<D��<#�
;�`B;ě�;D��:�o��o��o�ě��ě���`B�t��#�
�#�
�D���D���T���e`B��C���t���t����㼣�
���
���
���
��1��1��9X��9X��9X��j��j��j���ͼ��ͼ�����������������/��/��`B��`B��`B���o�C��C��t���P��P��P��P��P��P�#�
�''8Q�8Q�<j�@��L�ͽe`B�m�h�q���u��O߽��
���1��1��1��-� $���������	
#/6/.21/(#		��������������������55BN[girgg[UNB54-+-5���	
	���������������������������������������������#<bs�������n0
���CHTXWTTLHD><CCCCCCCC^ahlmvz��������zmaY^*;HTamz�����mTH<-'%*�������������������������������������������������������������)5BB/#�����������������������������!"!
����������������������������$)6660)���  ���������������� 

�������������NN[bgtw����tg[ZNNKNNru�������������urmkr��������������������>BBNW[][WPNKB<;:>>>>T[g������������gdVQTrtv�������������xtrr��������������������~������������������~������������������������������������������������������'6COTXXTLC6* �^am|����������zmc]\^yz������������~ztsuy������������������������������������������������������������5NZXTSOB5)
��������������������.6<BDLMJHLBA64**....MO[h�����������h[QQM067BORPSOHFB651/0000<BOX[ghnkhd[OB@::<<<PTaz�����������pbRLPjq����������������hj��������������������!#/;;/%#!!!!!!!!!!!������������������#Haz�����nH<##01TUahjida]URJLOTTTTTT0Ibnruv{}~{snUI@820sy���������������tms����������������������������������)6Bht|��~{qqh[WVO?-)������������������������������������Uanz��zna]UUUUUUUUUU##0<>><70--#########8<IU_ba_YUI=82115<?8��������������������������������X[hhlkhb[XXXXXXXXXXX)<IUY^VQNIA<0/)(*+'):BIO[hjmjhb[OKB9::::���������������NOPT[hqtqljhh[SOMMNN�����������������
������qw������������}tnkq���
#/78<BE</#
����##/2//.,,.#8<<CHKUaaea^UNH<8888����������������������  &$�����pt�������������xtphp�ֺͺкѺֺ�������ֺֺֺֺֺֺֺֽ��ݽнĽŽѽݽ���������������������!�-�9�-�!���������������������������������������������������
���"�/�0�4�/�#�"����������
���������
��#�-�/�<�<�<�;�/�#��������|�~����������������������������������g�O�:�(�����N���������������������������������������������������������ĿĳĦĚĖčĊčĜĦĳĿ��������������Ŀ����������������������/�5�%���	�������0�$�������ƹƻ��������$�0�8�G�J�E�=�0àÖÌÆ�z�v�s�s�zÇÓàö����������ùàÇÂÂÇÓàì÷÷ìàÓÇÇÇÇÇÇÇÇ�H�;�2�/�8�6�"�"��&�;�T�a�z�������~�m�H�U�L�H�C�D�H�U�\�a�h�n�p�n�a�U�U�U�U�U�U�Z�L�A�5��(�5�N�g�s�x���������}�w�s�g�Z�f�b�b�g�s�w�������ʾϾʾƾ���������s�f�f�f�Z�Y�Z�f�k�f�[�e�f�g�s�|�{�u�s�s�f�fìáéìù��������ùìììììììììì������)�0�)�)������������������������������������������������������������������������ʾо־ھ���ʾ���ƳƪƯƳƼ��������������ƳƳƳƳƳƳƳƳ������������������������������������������۾ܾپԾ׾��	��.�;�G�T�D�+�"�	�����	�������������������&� ���������������������ʾ��������߾ھ׾ʾ��U�I�<�2�7�G�U�{ŇŚŭŹ��ŹųŤŔ�n�b�U���ܻٻܻ����@�Y�f�t�y�u�h�Y�M�'��`�T�G�;�'��.�;�G�T�m�y�|�~�{�u�s�p�m�`�m�`�Q�F�@�?�A�F�G�T�y���������������y�m�;�.�"����������	��"�*�1�3�>�H�G�;��ƹƣƚƁ�wƄƎƚƧƳ���������������������������������������������������������������������������������������`�Z�^�`�d�m�y�����������y�m�`�`�`�`�`�`���|���������������������������������������������*�C�O�\�n�s�q�h�\�O�C�6�*����|�������������������������������������F�;�:�8�:�F�S�_�l�x�{�x�n�l�_�S�F�F�F�F���|����t�p�k�l�x�����ûл̻������������.�"�"���"�.�;�A�G�T�\�U�T�G�;�.�.�.�.�m�c�d�m�p�y�z���������������������y�m�m�ѿſ������ʿݿ���(�A�N�R�R�J�5�����Ѻ��~�L�'����'�3�F�L�e�r���������������ܹչܹ������������ܹܹܹܹܹܹܹܾM�L�G�M�Z�b�f�g�f�Z�M�M�M�M�M�M�M�M�M�M�����)�/�5�B�F�F�B�=�5�)�������:�1�'�$�+�Z�m�������������������z�m�O�:��������	��"�)�.�4�.�"��	�������������,�$�$�*�-�:�F�S�_�l�x���x�t�x�s�_�F�:�,��ŹŔ�{�}ŇŔşŭŹ������������������ƻ��������������������������������'�"����&�4�@�M�Y�^�f�k�m�m�f�Y�@�4�'���z�w�}����������������"�+��	���������L�J�D�E�H�L�X�e�r�y�|�~�����~�r�e�\�Y�L�Ǻĺ��������������������ɺкֺغֺѺ̺��H�E�F�F�H�U�W�Y�W�U�H�H�H�H�H�H�H�H�H�H���������������ĽʽĽ��������������������н˽ս�����(�4�:�?�4�(������߽ݽ���������ýþ����������5�)���������һ������������������������ûʻлһлŻû���޹���������������������л������ûл�����'�/�,�'������ܻн��ݽ׽ҽٽݽ�����������������ā�yāčĖĚĦĮĳļĳĦĚčāāāāāā�����������������������ʼ˼ԼʼǼ��������$����#�(�,�0�=�I�P�V�Y�V�N�I�=�0�,�$�j�`�T�f�l���������ĽнڽнĽ��������y�j�/�#�����#�/�<�H�R�U�a�b�X�U�J�H�<�/D�D�D�D�D�D�D�D�D�EEEEE E*E2E-EED�E�E�E�E�E�E�E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�EzE�E�E�E�E�E�E�E�E�E�E�E�E����߼������������� ����������� ���������!�!�(�#�!�����ĦġğĚĚĥĦĳĿ����������������ĿĳĦ L F � / ) ) @ F Q - 4 P Y f i Q G c y M b ` ] Z 8 F ? < p ^ E H y Z W a - H / V B S F & G = ` O U Y O A 7 <  _ - ` � F I 7 r Y Y 5 x U M u 5 / k b O a ,    �  M  Q  �  �  *  �  �  |  �  �  k  u  �  �  �  �    �  >  1  .  �  a  �  
  !  �  |  J  -  _  �  �  R  �  �  �  �  X  �  ]  �  G  #  =  J  ?  �  �  �  F  K  u  �  �  I  5  �  @  �  �      !    �  (  j    w  O     �    �  %<�t��D��<o���
;o�D�����
����#�
��w�aG��<j�t���o�ixռ���+������o���
��9X��1�o��9X��j�aG�����%�������49X�<j�#�
�49X��P��h��h������`B�8Q콗�P��P�']/��G��C��0 Ž�w��o�#�
�L�ͽy�#�<j�� Ž��T�D���<j�<j�@�������vɽ}�L�ͽixսy�#������t�������Q�t��������;)��ȴ9��^5B#�B�7B��BA���B�QB5�B'u�A���B A��IBk�B|BxcB�/B �=BqB!��B��BkB��B	B�B4�B)�B@B	�B
��B�5B�!B!.B�B+�B/��A���B N�BդB��BzB�Bs�B�B%�B8�B�2B 
*B�bBΞB�B��B��B��B'b�BTwB!)B�B�LB!F[B#
�B�7B%��B&��B͈BPB��B&p�B��B�BpNB]IBEpB
�B'�B��Bp�BJBB
k�B|&B��B1�BA��NB��B9�B'�%A�{�B <9A��OB8wBb�B�rB�XB �B)QB!�XB��BáB��B	u�B4�EB?'B�UB	�B
��B�B�WB!@BD�B+@@B0@<A��3B @B�?BY�B��B?�BA�B�]B�ZB:YB�]B b�B�B�*BĻB�>B�8B�"B'>�B`!B��B�6BB!A�B#y�B��B%�4B&ĔB��B?�B�B&C�B?�B��B��B��B��B
�RB��B�4B�B��B@1B	�l@@A.��@juTA�D�A��A��AHhQA��A���AᣥA�jB�vA˱A�-QA�YA���A�߀AI-AA�UÄ́�AՍ�AJvAO��B��A�j�AZ�BA���AR�-A��@���Ag��Aj��A]X�Bx�AҀ�A�X�Ak�2@�ӎB H[Ap��@��W@��9Ab��Ao7HA��1@ �l?XA>�iA���A���A\��@�o9A�p@��@�"A�#Z?��@)kAĬQA$>�A3�4A�o�@���?+{�@��:A-�Aߊ@���B
u�A �A�C�T�C�o"C��AŴA
#�A��D@D"�A.�*@j��A���A�BA���AG!�A��eA��AᄀA��'B	�<Ȁ`A�|`A��mA�j�A�Y*AI AB��A�YA��AK�ANP�B�AA�ewAZ�gA�zASA�U@��AguAk!AA[@B�A��sA�~�Aj�c@��B ?�Ao9�@�-�@���Ab�Ao&zA���@V-?#M�A>��A�;6A��yA\~�@{��A�w@�/�@�	�A�K?��l@-p�A�A#$A4�BAу�@���?2D@���A-� Aߨf@��oB
�HA"g�A�|�C�I�C�j;C��A A
�A�~�                     	   g      $   2   )         0   
                              $      ,   -   Q                           _         1            S                        
   2   -      
         '   1               
               A         S                              C         )   )         1      !                           !      !   %   %         !                           '         )   /            ?         %         -                  !                                                                     7         #   %         !                                          %            !                                    )               ?         #         #                                                               N�T;N^�/M�)N��tN��}OP�N�PP�G
No��O?�O�2O�i]O��Num�O��RN��HN�=tOK�Nkq�N)�/M�
;N�|O'`�N%��N���Os;N��OHZXOʘOV��O�"�O���OfrbOy��O!N��_N�}'N��zOCW�N(ȳN�W�Oc��N��{O�P$cO�'N�N.�NO�QP��PN��On�O���NX��O\B�O��O�%N��9M�d�N�N��1Oa�mN�f	M�O?Ov�tN�R�N^b�N���OcORx�OP�OkB�N��N���O^M�N��NOʯ  �  ]  �  �  �  �  �  �  L  �  �  �  �    s  �  U  s  �  0  �  �  �  �  �  V    �  �  �  �  4  |  �  l    Z    c  �    �  �  �  �  �  X  �  �  ]  �  s  �  
  	F  �  d  �  N  �    	A  �      q  �  �  �  s  /  3  0  �  3  �  |<ě�;ě�<D��%   ;�`B;ě�;o���㻃o�o��t��t��t��t��ě��#�
���ͼu�T���e`B��C���t����
���㼣�
�t����
�+��1�<j��9X��j��9X������j��j���ͼ��ͽ��������/�,1��/��`B��`B�}��`B���+�C��C���P�,1��P�,1�49X��P��P�,1�'Y��ixս<j�<j�@��L�ͽe`B�u�q���u��\)��E����1��-��{��-� $���������#(,(#��������������������257BN[d^[TNB54222222���	
	���������������������������������� �����������#0I{�������nI0CHTXWTTLHD><CCCCCCCCjmnz����������zmjcdj/6;T`msx{|{vmaT:0+*/��������� ������������������� �������������������������������������������������������������������

 ����������������������������$)6660)���  ���������������� 

�������������NN[bgtw����tg[ZNNKNNou~�������������vuqo��������������������>BBNW[][WPNKB<;:>>>>dgt���������tqgg`_ddrtv�������������xtrr��������������������~������������������~������������������������������������������������������'6COTXXTLC6* �`gm�����������zmh`^`yz������������~ztsuy������������������������������������������������������������)5BHKJIDB85)&��������������������/6ABCKLIGDBB64*+////bhs������������tjh^b067BORPSOHFB651/0000?BOV[ehljhb[OBB;;=??PTaz�����������pbRLP����������������������������������������!#/;;/%#!!!!!!!!!!!��������������������#Haz�����nH<##01TUahjida]URJLOTTTTTT2IUbnqtt{~{nbUID<932r}���������������ztr������������������������ ����������/6B[huz|xrkh[XOLB2//������������������������������������]anz|zna`]]]]]]]]]]##0<>><70--#########;<@IUV[]][USIE><;:;;�����	��������������������������X[hhlkhb[XXXXXXXXXXX)<IUY^VQNIA<0/)(*+'):BIO[hjmjhb[OKB9::::���������������NOQU[hpspjhh\[[ONMNN�����������������
������rtx������������~ttmr��
#-/126/
������##/2//.,,.#8<<CHKUaaea^UNH<8888����������������������&#�����pt�������������xtphp�ֺͺкѺֺ�������ֺֺֺֺֺֺֺֽ��ݽ�����
���������������������!�-�9�-�!���������������������������������������������������
���"�/�0�4�/�#�"����������
���������
��#�-�/�<�<�<�;�/�#��������~����������������������������������s�]�N�>�'�&�0�P�s�����������������������������������������������������������ĦěĚĎđĠĦĳĻĿ��������������ĿĳĦ���������������������	��"�"��	���������0�$�������������������$�0�7�F�H�C�=�0ùàØÓÎÇ�z�w�u�u�zÇÓàì��������ùÇÂÂÇÓàì÷÷ìàÓÇÇÇÇÇÇÇÇ�H�<�:�A�@�<�<�9�H�T�m�z�������z�l�a�T�H�U�L�H�C�D�H�U�\�a�h�n�p�n�a�U�U�U�U�U�U�g�d�Z�T�N�L�G�N�Z�c�g�s�u�v�s�k�g�g�g�g�s�l�n�s������������Ⱦ¾�������������s�f�f�Z�Y�Z�f�k�f�[�e�f�g�s�|�{�u�s�s�f�fìáéìù��������ùìììììììììì������)�0�)�)��������������������������������������������������������������������������ʾ;Ӿ۾ݾ׾վʾ�ƳƪƯƳƼ��������������ƳƳƳƳƳƳƳƳ����������������������������������������������������	���"�+�&�"���	�������	�������������������&� �����������������ƾʾ׾����� �����پʾ��U�I�<�2�7�G�U�{ŇŚŭŹ��ŹųŤŔ�n�b�U�4�'��� �����'�4�@�M�Y�a�f�l�h�f�Y�4�`�T�G�;�'��.�;�G�T�m�y�|�~�{�u�s�p�m�`�m�`�R�G�A�@�C�G�T�m�y���������������y�m�;�.�"����������	��"�*�1�3�>�H�G�;��ƳƦƚƎƆƎƚƧƳ�����������������������������������������������������������������������������������������`�Z�^�`�d�m�y�����������y�m�`�`�`�`�`�`���|�������������������������������������*�!������*�6�C�O�T�\�^�^�\�P�C�6�*���|�������������������������������������F�<�:�9�:�F�S�_�l�x�y�x�m�l�_�S�F�F�F�F�������~�{�y�����������������������������.�"�"���"�.�;�A�G�T�\�U�T�G�;�.�.�.�.�m�g�f�m�q�y�|���������������������y�m�m�ѿſ������ʿݿ���(�A�N�R�R�J�5�����Ѻ~�r�e�T�L�B�H�O�Y�e�r�����������������~�ܹչܹ������������ܹܹܹܹܹܹܹܾM�L�G�M�Z�b�f�g�f�Z�M�M�M�M�M�M�M�M�M�M�����)�5�5�B�B�D�B�:�5�)�������:�1�'�$�+�Z�m�������������������z�m�O�:��������	��"�)�.�4�.�"��	�������������-�%�$�%�*�-�:�F�U�_�k�o�r�r�n�_�O�F�:�-��ŹũŠńŉŔŠŨŭŹ�����������������߻��������������������������������1�'�"�!�'�*�4�@�M�[�f�i�k�k�f�c�Y�M�@�1�����}���������������������������������L�J�D�E�H�L�X�e�r�y�|�~�����~�r�e�\�Y�L�Ǻĺ��������������������ɺкֺغֺѺ̺��H�G�H�G�H�U�V�X�U�U�H�H�H�H�H�H�H�H�H�H���������������ĽʽĽ���������������������������������(�3�4�8�4�(�!��������������������������������������޻����������������������ûɻлѻлĻû�����޹���������������������л������ûл�����'�/�,�'������ܻн��ݽ׽ҽٽݽ�����������������ā�yāčĖĚĦĮĳļĳĦĚčāāāāāā�������������������������ʼ˼ʼ¼��������$����#�(�,�0�=�I�P�V�Y�V�N�I�=�0�,�$�j�`�T�f�l���������ĽнڽнĽ��������y�j�/�,�#�����#�/�<�H�P�U�a�W�U�H�G�<�/D�D�D�D�D�D�D�D�D�EEE%E-E*E#EEED�D�E�E�E�E�E�E�E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�EzE�E�E�E�E�E�E�E�E�E�E�E�E���������������������������������������!�'�"�!�����ĦġğĚĚĥĦĳĿ����������������ĿĳĦ L 4 � 8 ) ) A 4 Q # + R [ f \ Q M [ y M b ` b Z 8 ) ? ' p R E D y T W a - H % V @ M F % G * ` O M Y O 5 < <  C - ` w F @ & h Y Y 5 x R M u 5 & k b L S ,    �  _  Q  �  �  *  �  c  |  �  �    K  �    �  �  [  �  >  1  .  �  a  �  7  !  �  |  �  -  (  �  %  R  �  �  �  �  X  �  �  �  *  #  0  J  ?  �  �  �  �  �  u  �  �  I  5  L  @  �  �  �    !    �  �  j    S  �     �  �  �  %  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  @g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  X  =       �  +  2  3  0  ,  '  B  U  [  U  M  @  .    �  �  4  �  E   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  N  "  �    D  r  �  �  �  �  �  �  �  �    S  1    �  /  �  �  l  �  �  �  �  �  �  �  z  j  W  E  2      �  �  �  �  �  p  �  �  �  �  �  �  m  M  )  �  �  �  Y    �  V  �  �  o   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  x  t  q  n  y  �  �  �  �  �  �  _  6  	  �  �  B  �  o  �  m  �  �    L  >  0  "           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  L     �  �  l  )  �  �     &  :  <  Z  �  �  �  �  �  �  �  {  E    �  a    �  H    �  �  i  q  �  �  �  �  q  Q  ,    �  �  U    �  �  :  �  Y  k    �  �  �  �  �  �  �  y  _  O  C  4      �  �  ^    r  �    �  �  �  �  �  �  �  |  J    �  �  �  ]  7    �  �  �  �        J  b  r  n  c  G     �  �  1  �  )    {  �  �  �  �  �  �  �  �  �  x  k  ]  M  ;  '      �  �  t     �  �  �  �       
        <  K  S  S  I  .    �  �  X    h  ^  U  V  q  p  k  d  ]  M  7    �  �  �  Z     �  �  b  �  �  �  �  |  n  _  M  8  #    �  �  �  �  �  �  �  �  �  0  +  &  !        �  �  �  e     �  �    \  9     �   �  �  �  ~  z  y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  s  k  c  [  W  S  P  M  J  F  A  6  ,  !        �  �  �  �  �  �  �  �  �  �  r  K    $  &    �  �  �  f  �  }  o  `  R  D  6  *           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  r  j  b  Y  Q  N  V  ]  e  l  t  {  �     }  �  �    /  E  S  T  J  2    �  z  1  �  }  �  a      �  �  �  �  �  �  �  �  �  �  �  w  d  Q  "  �  �  5    u  �  �  �  �  �  �  �  �  �  �  i  )  �  m  �  M  �  �  �  �  �  �  K  �  p  =    �  �  G    �  g    �  �  �  d  �  
2  
�  -  g  �  �  ~  f  >    
�  
P  	�  	L  �  �  �    B  �  �  �  x  ^  =    �  �  �  U  $  �  �  s  )  �  /  �  ^  +  4  0  *  "      �  �  �  �  n  D    �  �  I  �  \   �  |  h  R  p  h  ^  W  H  3      �  �  �  u  /  �  �     �  �  �  �  �  �  �  �  l  J  -    �  �  �  D  �  �  H  �  8  l  T  =  &    �  �  �  �  �  �  �  F  �  �  B  �    $  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  M  @  2  %    	  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  b  A    �  �  �  Q  �  E  �  �  6  	�  
S  
�  �  �    C  Y  c  W  ;  	  �  @  
�  	�  �  �  �     �  �  �  �  }  {  y  v  t  q  p  q  r  s  t  u  v  w  x  y  
      �  �  �  �  �  s  _  N  6    �  �  �  �  u  [  3  �  :  V  e  s  x  �  |  V  )  �  �  ;  �  �  %  �    i  �  �  �  �  �  �  �  �  �    u  h  X  A  '    �  �  �  {  V  �  �  �  �  �  �  �  �  �  �  �  y  Z  9    �  �  �  |  K  �  �  �  u  `  E  #     �  �  �  x  R    �  �  N  �  ~   �    @  A  n    �  �  �  �  �  �  S    �  c  �  +  "  �  �  X  R  L  G  @  :  4  .  '  !        �  �  �  �  �  �  �  �  �  �  �  r  X  =    �  �  �  �  L    �  �  k  B    �  �  �  �  �  �  �  �  �  �  ~  h  R  :  "    �  �  �  U    ]  K  6    �  �  �  �  �  �  �  y  C    �  j  �  �  {  v  �  �  �  �  �  �  �  �  �  �  s  f  X  I  ;  ,     #  '  *  M  i  h  Y  I  8  $    �  �  �  �  �  �  �  z  Z    �  	  �  �  �  �  �  �  �  �  �  m  Z  M  =    �  �  r  $  �  �  
  �  �  �  �  �  j  V  B  /      ,  6  2  ,      �  �  	  	5  	E  	;  	'  	  �  �  �  R    �  D  �  m  �  W  l  �  �    @  }  �  �  |  n  Z  ?     �  �  �  l    �  $  �  R  �  d  R  A  1  !      �  �  �  �  �  x  T  0  
  �  �  �  �  �  z  n  d  Z  P  F  ;  0  "      �  �  �  �  �  x  Y  9  *  1  7  >  C  H  M  P  R  T  R  K  D  8      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  C     �  �  �  �  ]  4    �  �  �  
        �  �  �  �  ^  1  �  �  �    Y  �  	+  	!  	  �  �  	=  	"  �  �  �  W  �  _  �  4  �  �  ^  W  N  �  �  �  �  q  O  0    �  �  �  W    �  �  F  �  �  Y  �    
  	        �  �  �  �  �  �  �  �  �    ?     �  �          �  �  �  �  �  �  �  s  [  A  "    �  �  X   �  q  p  m  e  [  Q  C  4  !    �  �  �  �  �  b  >    �  �  �  �  �  �  n  ]  K  9  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  9  �  �  l  Q  I  =    �  �  F  �  �  �  �  i  >    �  �  �  �  �  n  >    �  �  P  	  �  a   �   T  s  Y  >  '    
  �  �  �  �  i  c  a  _  b  f  t  p  Z  =  $  .  (        �  �  �  �  Q    �  �  V    �  %  �   �  
�    *  -    
�  
�  
�  
z  
Q  
  	�  	�  	(  �  �  $  U  �  i  0  �  �  �  [  P  >  �  �    K    �  �  �  S  �  O  �  �  �  `  5  �  �  �  j  k  F    �  �  �  w  8  �  ?  �  $  �  +  1  "  
  �  �  i    �    ~  �  !  _  �  
�  	  8    �  �  �  �  �  p  O  -  	  �  �  �  [  (  �  �  �  �  @  �  3  |  Y  6    �  �  �  �  �  }  m  `  S  F  9  (       �   �